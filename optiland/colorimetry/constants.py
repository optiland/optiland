"""Standard colorimetric data (CIE).

Data are sampled at 1 nm from 380 nm to 780 nm.
Values are frozen in a local JSON file generated from official CIE datasets,
so runtime does not depend on external colorimetry libraries.
"""

from __future__ import annotations

import json
from pathlib import Path

_DATA_PATH = Path(__file__).with_name("colorimetric_data_1nm.json")
_DATA = json.loads(_DATA_PATH.read_text())

WAVELENGTHS_STD = [int(wl) for wl in _DATA["WAVELENGTHS_STD"]]
ILLUMINANT_D65 = [float(v) for v in _DATA["ILLUMINANT_D65"]]

CIE_1931_2DEG = [tuple(float(v) for v in row) for row in _DATA["CIE_1931_2DEG"]]
CIE_1964_10DEG = [tuple(float(v) for v in row) for row in _DATA["CIE_1964_10DEG"]]
