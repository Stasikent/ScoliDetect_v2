# -*- coding: utf-8 -*-
from __future__ import annotations
import csv
from pathlib import Path
import argparse

def main():
    ap = argparse.ArgumentParser(description="Создать labels.csv из to_label_manifest.csv")
    ap.add_argument("--manifest", type=str, default="to_label_manifest.csv", help="файл манифеста")
    ap.add_argument("--out", type=str, default="labels.csv", help="куда записать шаблон разметки")
    args = ap.parse_args()

    man = Path(args.manifest)
    if not man.exists():
        raise SystemExit(f"manifest not found: {man}")

    rows = []
    with man.open("r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append({
                "file": r["file"],          # абсолютный/относительный путь к DICOM
                "true_angle_deg": "",       # ← здесь вы руками вписываете измеренный угол (число)
                "notes": ""                 # опционально
            })

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["file", "true_angle_deg", "notes"])
        wr.writeheader()
        for r in rows:
            wr.writerow(r)

    print(f"template saved: {args.out} (rows={len(rows)})")

if __name__ == "__main__":
    main()
