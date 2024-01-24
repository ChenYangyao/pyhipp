from __future__ import annotations
from .abc import mpl_colors, plt, MplObj, mpl_cm
from typing import Tuple, Any, Union, Self
import numpy as np

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
    
class Normalize(MplObj[mpl_colors.Normalize]):
    pass
        
class Colormap(MplObj[mpl_colors.Colormap]):    
    
    Raw = mpl_colors.Colormap
    
    def __call__(self, x: np.ndarray, alpha=None):
        return self._raw(x, alpha=alpha)
        
class ScalarMappable(MplObj[mpl_cm.ScalarMappable]):
    
    Raw = mpl_cm.ScalarMappable
    
    @classmethod
    def from_cmap(cls, cmap: Colormap, norm: Normalize) -> Self:
        raw = cls.Raw(cmap=cmap._raw, norm=norm._raw)
        return cls(raw)
    
_predefined_color_seqs = {
    'dark2': plt.get_cmap('Dark2').colors,
    'tab10': plt.get_cmap('tab10').colors,
    'set1': plt.get_cmap('Set1').colors,
    'set2': plt.get_cmap('Set2').colors,
}
    
class ColorSeq:
    def __init__(self, colors: list[Color.ColorSpec]) -> None:
        self.colors = list(Color(c) for c in colors)
        
    @staticmethod
    def predefined(name: str = 'dark2') -> ColorSeq:
        colors = _predefined_color_seqs[name]
        return ColorSeq(colors)
        
    @staticmethod
    def from_cmap(cmap: str | Colormap, levels: list[float] | int, alpha=None):
        if isinstance(cmap, str):
            cmap = Colormap(plt.get_cmap(cmap))
        if isinstance(levels, int):
            levels = np.linspace(0., 1., levels)
        return ColorSeq(cmap(levels, alpha=alpha)) 
        
    def get_rgba(self) -> list[Color.RgbaSpec]:
        return [c.get_rgba() for c in self.colors]

_predefined_color_seqs |= {
    'rainbow_5': ColorSeq.from_cmap('rainbow', 5).get_rgba(),
    'rainbow_10': ColorSeq.from_cmap('rainbow', 10).get_rgba(),
    'gnuplot_5': ColorSeq.from_cmap('gnuplot', 5).get_rgba(),
    'gnuplot_10': ColorSeq.from_cmap('gnuplot', 10).get_rgba(),
    'gnuplot2_5': ColorSeq.from_cmap('gnuplot2', 5).get_rgba(),
    'gnuplot2_10': ColorSeq.from_cmap('gnuplot2', 10).get_rgba(),
}


    
    