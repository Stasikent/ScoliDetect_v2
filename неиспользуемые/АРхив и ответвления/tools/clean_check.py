# tools/clean_check.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, sys, ast, re, shutil, time
from pathlib import Path
from typing import Dict, Set, List, Tuple

PROJECT_ROOT_DEFAULT = Path(__file__).resolve().parents[1]

PY_IGNORES = {
    "__init__.py",  # всегда считаем служебным
    # добавь сюда файлы, которые хочешь исключить из «мусора»
}

ENTRYPOINT_GLOBS = [
    "thor_scoli_app/cli.py",
    "cobb_*.py",
    "label_gui.py",
    "select_for_labeling.py",
    "train_and_apply_correction.py",
]

ARTIFACT_DIR_HINTS = ["out", "overlays", "reports", "to_label", "results", "models", "logs"]

REQ_FILE = "requirements.txt"
REPORT_FILE = "project_audit_report.txt"


def _is_python_module(p: Path) -> bool:
    return p.suffix == ".py" and p.name not in PY_IGNORES and "venv" not in p.parts

def _pkg_name_for(path: Path, root: Path) -> str:
    rel = path.resolve().relative_to(root.resolve())
    parts = list(rel.parts)
    if parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]
    # модульное имя вида thor_scoli_app.auto_cobb
    return ".".join(parts)

def _imported_modules_in_file(py_path: Path) -> Set[str]:
    """Собирает строки импортов: top-level import & from import."""
    out: Set[str] = set()
    try:
        src = py_path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(src, filename=str(py_path))
    except Exception:
        return out
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                out.add(n.name.split(".")[0])
                out.add(n.name)  # и полные
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                # относительный импорт: from .auto_cobb import X
                dots = "." * node.level if getattr(node, "level", 0) else ""
                out.add(dots + node.module)
            else:
                # from . import something
                if getattr(node, "level", 0):
                    out.add("." * node.level)
    return out

def _collect_all_py(root: Path) -> List[Path]:
    return [p for p in root.rglob("*.py") if _is_python_module(p)]

def _normalize_import_to_path(imp: str, where: Path, root: Path) -> List[Path]:
    """
    Пытается перевести запись импорта в путь файла(ов).
    Возвращает список возможных путей (module.py или package/__init__.py)
    """
    out: List[Path] = []
    if imp.startswith("."):
        # относительный импорт
        base = where.parent
        # посчитать уровень
        level = len(imp) - len(imp.lstrip("."))
        base = base
        for _ in range(level):
            base = base.parent
        tail = imp[level:]
        if tail:
            parts = tail.split(".")
            candidate = base.joinpath(*parts)
        else:
            candidate = base

        # module.py
        mod = candidate.with_suffix(".py")
        if mod.exists():
            out.append(mod)
        # package/__init__.py
        init = candidate / "__init__.py"
        if init.exists():
            out.append(init)
        # м.б. from .pkg import sub; тогда проверим папку
        if candidate.is_dir():
            for c in candidate.glob("*.py"):
                out.append(c)
        return list(dict.fromkeys(out))  # uniq

    # абсолютный импорт внутри проекта
    parts = imp.split(".")
    candidate = root.joinpath(*parts)
    mod = candidate.with_suffix(".py")
    if mod.exists():
        out.append(mod)
    init = candidate / "__init__.py"
    if init.exists():
        out.append(init)
    return out

def _resolve_used_graph(py_files: List[Path], root: Path) -> Set[Path]:
    """Строит граф модулей, reachable из entrypoints."""
    by_name: Dict[str, Path] = {}
    for p in py_files:
        by_name[_pkg_name_for(p, root)] = p

    # стартовые файлы
    entry_files: Set[Path] = set()
    for pat in ENTRYPOINT_GLOBS:
        for p in root.glob(pat):
            if p.exists():
                entry_files.add(p.resolve())

    # если не нашли явных, возьмём все py в корне как потенциальные скрипты
    if not entry_files:
        entry_files = {p.resolve() for p in py_files if p.parent == root}

    used: Set[Path] = set()
    stack: List[Path] = list(entry_files)

    while stack:
        cur = stack.pop()
        if cur in used:
            continue
        used.add(cur)
        imps = _imported_modules_in_file(cur)
        for imp in imps:
            for target in _normalize_import_to_path(imp, cur, root):
                if target.exists() and target not in used:
                    stack.append(target)
    return used

def _find_artifacts(root: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".csv", ".pkl", ".json", ".txt", ".svg"}
    out: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            # не цепляем файл отчёта текущего запуска
            if p.name == REPORT_FILE:
                continue
            # подсказка по артефактным папкам
            if any(h in p.as_posix().lower() for h in ARTIFACT_DIR_HINTS):
                out.append(p)
    return out

def _load_requirements(req_path: Path) -> List[str]:
    if not req_path.exists():
        return []
    pkgs = []
    for line in req_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        pkgs.append(re.split(r"[=<>!~\[]", line, maxsplit=1)[0].strip().lower())
    return pkgs

def _guess_third_party_imports(py_files: List[Path], root: Path) -> Set[str]:
    """Грубо: имена импортов, не принадлежащие проекту (не резолвятся в файлы)."""
    names: Set[str] = set()
    for p in py_files:
        for imp in _imported_modules_in_file(p):
            if imp.startswith("."):
                continue
            # если не резолвится в файл проекта — считаем внешним
            if not _normalize_import_to_path(imp, p, root):
                names.add(imp.split(".")[0])
    return names

def analyze(root: Path, move_unused: bool=False) -> str:
    root = root.resolve()
    py_files = _collect_all_py(root)
    used = _resolve_used_graph(py_files, root)
    unused = sorted(set(py_files) - used)

    artifacts = _find_artifacts(root)
    reqs = _load_requirements(root/REQ_FILE)
    third_party = sorted(_guess_third_party_imports(py_files, root))

    not_in_requirements = [n for n in third_party if n.lower() not in reqs]
    maybe_unused_reqs = [r for r in reqs if r.lower() not in [t.lower() for t in third_party]]

    lines = []
    lines.append(f"# Project audit — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Root: {root}")
    lines.append("")
    lines.append("## Used Python files:")
    for p in sorted(used):
        lines.append(f"  - {p.relative_to(root)}")
    lines.append("")
    lines.append("## Unused Python files (candidates to archive):")
    if unused:
        for p in unused:
            lines.append(f"  - {p.relative_to(root)}")
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append("## Artifact-like files (images/csv/pkl/txt in report/output dirs):")
    if artifacts:
        for p in sorted(artifacts):
            sz = p.stat().st_size
            lines.append(f"  - {p.relative_to(root)}  ({sz/1024:.1f} KiB)")
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append("## Third-party imports (guessed from code):")
    if third_party:
        lines.append("  " + ", ".join(third_party))
    else:
        lines.append("  (none)")
    lines.append("")
    if reqs:
        lines.append("## requirements.txt present:")
        lines.append("  " + ", ".join(reqs))
    else:
        lines.append("## requirements.txt not found")
    lines.append("")
    if reqs:
        lines.append("## Imported but NOT listed in requirements.txt:")
        lines.append("  " + (", ".join(not_in_requirements) if not_in_requirements else "(none)"))
        lines.append("## Listed in requirements.txt but NOT imported in project:")
        lines.append("  " + (", ".join(maybe_unused_reqs) if maybe_unused_reqs else "(none)"))
        lines.append("")

    if move_unused and unused:
        dst = root / "archive_unused"
        dst.mkdir(exist_ok=True)
        for p in unused:
            rel = p.relative_to(root)
            target = dst/rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), str(target))
        lines.append(f"Moved {len(unused)} unused *.py files to archive_unused/")

    report_path = root / REPORT_FILE
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return str(report_path)

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Audit project for unused files and artifacts")
    ap.add_argument("--root", type=str, default=str(PROJECT_ROOT_DEFAULT), help="Project root")
    ap.add_argument("--archive-unused", action="store_true", help="Move unused *.py files into archive_unused/")
    args = ap.parse_args()

    report = analyze(Path(args.root), move_unused=args.archive_unused)
    print(f"[OK] audit report saved to: {report}")

if __name__ == "__main__":
    main()
