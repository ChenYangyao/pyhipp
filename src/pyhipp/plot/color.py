from __future__ import annotations
from .abc import mpl_colors, plt
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
    
_predefined_color_seqs = {
    'dark2': plt.get_cmap('Dark2').colors,
    'set1': plt.get_cmap('Set1').colors,
}
    
class ColorSeq:
    def __init__(self, colors: list[Color.ColorSpec]) -> None:
        self.colors = list(Color(c) for c in colors)
        
    @staticmethod
    def predefined(name: str = 'dark2') -> ColorSeq:
        colors = _predefined_color_seqs[name]
        return ColorSeq(colors)
        
    def get_rgba(self) -> list[Color.RgbaSpec]:
        return [c.get_rgba() for c in self.colors]
    