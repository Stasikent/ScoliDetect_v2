# tools/import_audit.py
# -*- coding: utf-8 -*-
import ast, os, argparse, json
from pathlib import Path
from collections import defaultdict

def list_py_files(root: Path):
    return [p for p in root.rglob("*.py") if ".venv" not in p.parts and "unused" not in p.parts]

def mod_name_from_path(root: Path, p: Path):
    rel = p.relative_to(root).with_suffix("")
    return ".".join(rel.parts)

def parse_imports(py_path: Path):
    text = py_path.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(text, filename=str(py_path))
    except Exception:
        return []
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                out.append(("import", n.name))
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            out.append(("from", mod))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", type=str)
    ap.add_argument("--out", default="project_imports_report.txt", type=str)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    py_files = list_py_files(root)

    # карта: модуль -> файл
    file_to_mod = {p: mod_name_from_path(root, p) for p in py_files}
    mod_to_file = {m: p for p, m in file_to_mod.items()}

    # граф импортов
    imports_by_file = {}
    refs_to_module = defaultdict(set)  # модуль -> кто импортирует
    for p in py_files:
        imports = parse_imports(p)
        imports_by_file[p] = imports
        for kind, name in imports:
            # интересны только внутренние модули проекта
            # нормализуем имя, чтобы поймать краткие формы:
            # - 'from foo.bar import x' -> 'foo.bar'
            # - 'import foo.bar' -> 'foo.bar'
            mod = name.split()[0]
            # пробуем сопоставить с локальными модулями
            # точное совпадение или префикс (пакеты)
            for local_mod in mod_to_file.keys():
                if mod == local_mod or mod.startswith(local_mod + "."):
                    refs_to_module[local_mod].add(file_to_mod[p])

    # найти файлы, которые никто не импортирует (входные точки исключаем при необходимости)
    entry_points = {
        "cli_ml", "train_and_apply_correction", "select_for_labeling",
        "label_gui", "train_border_ml", "check_labels_vs_results",
        "annotate_borders"
    }
    unused = []
    for p, m in file_to_mod.items():
        # __init__ пропускаем
        if p.name == "__init__.py":
            continue
        # если модуль — входная точка (запускается как скрипт), он может ни кем не импортироваться
        is_entry = any(m.endswith(ep) for ep in entry_points)
        if not refs_to_module.get(m) and not is_entry:
            unused.append((m, p))

    report = {
        "files_total": len(py_files),
        "imports_graph": {str(file_to_mod[p]): imports_by_file[p] for p in py_files},
        "refs_to_module": {k: sorted(v) for k, v in refs_to_module.items()},
        "unused_modules": [(m, str(p)) for m, p in unused],
        "hint": "unused_modules — кандидаты на перенос в unused/."
    }

    out = Path(args.out)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] report saved: {out}")

if __name__ == "__main__":
    main()
