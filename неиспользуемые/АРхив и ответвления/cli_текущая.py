# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np
import pydicom
import typer
from rich import print
from rich.progress import track

from .auto_cobb import auto_cobb_from_gray, CobbResult


app = typer.Typer(add_completion=False, no_args_is_help=True)


def _read_dcm_to_u8(path: Path) -> np.ndarray:
    """Читает DICOM и возвращает 8-битное grayscale с учётом инверсии."""
    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array.astype(np.float32)

    # нормализация
    arr -= float(arr.min())
    mx = float(arr.max())
    if mx > 0:
        arr /= mx
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)

    # DICOM может быть MONOCHROME1 (чёрное и белое поменяны местами)
    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        arr = 255 - arr
    return arr


def _ensure_dir(p: Optional[Path]) -> None:
    if p is None:
        return
    p.mkdir(parents=True, exist_ok=True)


def _save_overlay(dst_dir: Optional[Path], stem: str, overlay: Optional[np.ndarray], base: np.ndarray) -> None:
    if dst_dir is None:
        return
    _ensure_dir(dst_dir)
    if overlay is None:
        overlay = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        cv2.putText(overlay, "Cobb: n/a", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    out_path = dst_dir / f"{stem}_cobb.png"
    cv2.imwrite(str(out_path), overlay)


def _iter_dicoms(input_dir: Path) -> List[Path]:
    exts = (".dcm", ".dicom")
    return sorted([p for p in input_dir.rglob("*") if p.suffix.lower() in exts])


@app.command()
def cli(
    input_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True, help="Папка с DICOM"),
    mode: str = typer.Option("auto", "--mode", "-m", help="Режим (пока только 'auto')"),
    out: Path = typer.Option("results.csv", "--out", "-o", help="CSV с результатами"),
    save_overlays: Optional[Path] = typer.Option(None, "--save-overlays", help="Папка для сохранения оверлеев"),
    edge_mode: str = typer.Option("dual", "--edge-mode", help="Способ извлечения линий: 'dual' или 'center'"),
    debug: bool = typer.Option(False, "--debug", help="Отрисовывать служебные слои"),
):
    """
    Измеряет угол Кобба на DICOM-снимках (AP, грудной отдел).
    """
    files = _iter_dicoms(input_dir)
    print(f"[bold]Найдено файлов: {len(files)}[/bold]")
    if not files:
        raise typer.Exit(code=1)

    _ensure_dir(out.parent)
    _ensure_dir(save_overlays)

    rows = []
    for p in track(files, description="Измерение Cobb"):
        try:
            print(f"[dim]→ {p.name}: читаю и нормализую…[/dim]")
            gray = _read_dcm_to_u8(p)

            print(f"[dim]→ {p.name}: автоизмерение…[/dim]")
            res: CobbResult = auto_cobb_from_gray(gray, edge_mode=edge_mode, return_overlay=True, debug=debug)

            _save_overlay(save_overlays, p.stem, res.overlay_bgr, gray)

            rows.append({
                "file": str(p),
                "projection": "AP",
                "metric": "cobb",
                "method": f"auto/{edge_mode}",
                "angle_deg": round(float(res.angle_deg), 2),
                "error": "",
            })
        except Exception as e:
            print(f"[red][ERROR][/red] {p.name}: {e}")
            rows.append({
                "file": str(p),
                "projection": "AP",
                "metric": "cobb",
                "method": f"auto/{edge_mode}",
                "angle_deg": "",
                "error": str(e),
            })

    # запись CSV
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "projection", "metric", "method", "angle_deg", "error"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    ok = sum(1 for r in rows if r.get("angle_deg") not in ("", None))
    bad = len(rows) - ok
    print(f"[green]Готово:[/green] записано {len(rows)} строк в {out}.  Успешно: {ok}, ошибок: {bad}.")


if __name__ == "__main__":
    app()
