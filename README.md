# Thoracic Scoliosis DICOM App (v0)
Minimal, **fresh start** app to measure thoracic Cobb angle on AP DICOMs.

## Features
- Load single DICOM file or a whole folder (recursively).
- Basic windowing & contrast normalization.
- Rough spine ROI extraction (vertical projection & morphological operations).
- **Semi‑automatic** Cobb: click 2 endplates (top and bottom), 2 points per line → angle computed.
- **Automatic (beta)**: Hough-based endplate line proposals; picks most tilted pair inside thoracic ROI.
- Overlay output (PNG) + CSV with `filename, projection, region, metric, method, angle_deg`.

> This is a clean baseline for the new project. We can later plug in learned detectors or reuse parts from Iteration 1.

## Install
```bash
python -m venv .venv && source .venv/bin/activate  # (on Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

## Usage
Measure an entire folder (auto mode, AP thoracic):
```bash
python -m thor_scoli_app.cli measure /path/to/dicoms --mode auto --out results.csv --save-overlays out_imgs
```

Interactive semi-auto on a single DICOM:
```bash
python -m thor_scoli_app.cli interactive /path/to/file.dcm --out out_single.csv --save-overlay out_single.png
```

## Notes
- This v0 assumes **AP thoracic** X-rays. If the DICOM contains full spine, the ROI finder tries to crop to thoracic region heuristically.
- If auto fails or seems off, use `interactive` mode to click lines precisely.
