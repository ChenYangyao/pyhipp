from __future__ import annotations
import typing
from typing import Self

def dms_to_deg(dms: str| tuple[float,float,float]) -> float:
    if isinstance(dms, str):
        dms = tuple(float(v) for v in dms.split(':'))
    d, m, s = dms
    deg = d + m / 60. + s / 3600.
    return deg

def hms_to_deg(hms: str| tuple[float,float,float]) -> float:
    if isinstance(hms, str):
        hms = tuple(float(v) for v in hms.split(':'))
    h, m, s = hms
    deg = (h + m / 60. + s / 3600.)*15.
    return deg