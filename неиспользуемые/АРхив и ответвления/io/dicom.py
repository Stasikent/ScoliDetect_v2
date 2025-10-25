import os
from typing import Tuple, Dict, Any
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut

def load_dicom(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load DICOM from path and return (image_float32, meta)."""
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array

    # Apply modality and VOI LUT if present
    try:
        arr = apply_modality_lut(arr, ds)
    except Exception:
        pass
    try:
        arr = apply_voi_lut(arr, ds)
    except Exception:
        pass

    arr = arr.astype(np.float32)
    # Normalize to [0,1]
    mn, mx = float(np.min(arr)), float(np.max(arr))
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    else:
        arr[:] = 0.5

    meta = {
        "Path": path,
        "PatientID": getattr(ds, "PatientID", None),
        "StudyDate": getattr(ds, "StudyDate", None),
        "SeriesDescription": getattr(ds, "SeriesDescription", None),
        "ViewPosition": getattr(ds, "ViewPosition", None),
        "BodyPartExamined": getattr(ds, "BodyPartExamined", None),
        "Rows": int(getattr(ds, "Rows", arr.shape[0])),
        "Columns": int(getattr(ds, "Columns", arr.shape[1]))
    }
    return arr, meta
