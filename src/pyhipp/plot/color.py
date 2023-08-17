from __future__ import annotations
from .abc import mpl_colors
from typing import Tuple, Any, Union

class Color:
    
    RgbaSpec = Tuple[float, float, float, float]
    RgbSpec = Tuple[float, float, float]
    StrSpec = str
    ColorSpec = Union[RgbaSpec, RgbSpec, StrSpec, 'Color']
    
    def __init__(self, c: ColorSpec = 'k', a: float = None) -> None:
        self._rgba = Color.to_rgba(c)
        if a is not None:
            self.alpha(a)
    
    def alpha(self, value) -> Color:
        self._rgba = (*self._rgba[:3], value)
        return self
    
    def get_alpha(self) -> float:
        return self._rgba[3]
    
    def get_rgba(self) -> RgbaSpec:
        return self._rgba
    
    @staticmethod
    def to_rgba(c: ColorSpec) -> RgbaSpec:
        if isinstance(c, Color):
            rgba = c._rgba
        else:
            rgba = mpl_colors.to_rgba(c)
        return rgba
    
    