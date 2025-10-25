# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import pydicom
import typer
from rich import print
from rich.progress import track

from .auto_cobb import auto_cobb_from_gray, CobbResult


app = typer.Typer(add_completion=False, no_args_is_help=True)


def _iter_dicoms(input_dir: Path) -> List[Path]:
    exts = (".dcm", ".dicom")
    return sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in exts)


def _read_dcm(path: Path) -> np.ndarray:
    """Читает DICOM и возвращает uint8 grayscale."""
    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array.astype(np.float32)

    arr -= np.nanmin(arr)
    m = float(np.nanmax(arr))
    if m > 0:
        arr /= m
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)

    if getattr(ds, "PhotometricInterpretation", "").upper() == "MONOCHROME1":
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
    cv2.imwrite(str(dst_dir / f"{stem}_cobb.png"), overlay)


@app.command()
def measure(
    input_dir: Path = typer.Argument(..., exists=True, dir_okay=True, file_okay=False, readable=True,
                                     help="Папка с DICOM"),
    mode: str = typer.Option("auto", "--mode", "-m", help="Режим (сейчас доступен только 'auto')"),
    out: Path = typer.Option("results.csv", "--out", "-o", help="CSV с результатами"),
    save_overlays: Optional[Path] = typer.Option(None, "--save-overlays", help="Папка для размеченных PNG"),
):
    """Измеряет угол Кобба на DICOM-снимках в прямой проекции (грудной отдел)."""
    if mode.lower() != "auto":
        print("[yellow]Пока доступен только режим 'auto'. Использую auto.[/yellow]")

    files = _iter_dicoms(input_dir)
    if not files:
        print(f"[red]В папке {input_dir} не найдено .dcm файлов[/red]")
        raise typer.Exit(code=1)

    _ensure_dir(out.parent)
    _ensure_dir(save_overlays)

    rows = []
    print(f"[bold]Найдено файлов: {len(files)}[/bold]")

    for p in track(files, description="Измерение Cobb"):
        try:
            gray = _read_dcm(p)
            res: CobbResult = auto_cobb_from_gray(gray, return_overlay=True)
            angle = float(res.angle_deg) if res.angle_deg is not None else None

            _save_overlay(save_overlays, p.stem, res.overlay_bgr, gray)

            rows.append({
                "file": str(p),
                "projection": "AP",
                "metric": "cobb",
                "method": "auto",
                "angle_deg": None if angle is None else round(angle, 2),
                "error": "",
            })
        except Exception as e:
            print(f"[red][ERROR][/red] {p.name}: {e}")
            rows.append({
                "file": str(p),
                "projection": "AP",
                "metric": "cobb",
                "method": "auto",
                "angle_deg": None,
                "error": str(e),
            })

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "projection", "metric", "method", "angle_deg", "error"])
        writer.writeheader()
        writer.writerows(rows)

    ok = sum(1 for r in rows if r.get("angle_deg") is not None)
    bad = len(rows) - ok
    print(f"[green]Готово:[/green] записано {len(rows)} строк в {out}.  Успешно: {ok}, ошибок: {bad}.")


if __name__ == "__main__":
    app()
