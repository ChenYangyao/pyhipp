from __future__ import annotations
from typing import Any, Union, Iterator, Iterable
from matplotlib.figure import FigureBase
from .abc import MplObj, mpl_figure, Artist
from .artist_formatter import SubplotsFormatter, SubFiguresFormatter
from .axes import Axes, AxesArray
from ..core.abc import HasSimpleRepr
import numpy as np
from .color import ScalarMappable

class FigureBase(MplObj[mpl_figure.FigureBase]):
    
    Raw = mpl_figure.FigureBase
    
    def __init__(self, raw: Raw = None, **kw) -> None:
        
        super().__init__(raw, **kw)
        
        self._last_draw = []        
        
    @property
    def last_draw(self) -> Artist:
        return Artist(self._last_draw[-1])
        
    def subplots(self, n = 1, share = False, extent = None, space = None, 
            ratios = None, mpl_subplot_kw=None, margin = None,
            **mpl_gridspec_kw):
        
        kw = SubplotsFormatter(n=n,share=share,extent=extent,space=space, 
            ratios=ratios, margin=margin).get_subplots_kw()
        if mpl_subplot_kw is not None:
            kw['subplot_kw'] |= mpl_subplot_kw
        kw['gridspec_kw'] |= mpl_gridspec_kw
        
        out_n = kw['nrows'] * kw['ncols']
        out = self._raw.subplots(**kw)
        if out_n == 1:
            out = Axes(out)
        else:
            out = AxesArray(out)
        
        return out
    
    def subfigures(self, n = 1, space = None, ratios = None, 
                   **mpl_subfigures_kw):
        kw = SubFiguresFormatter(n=n, space=space, ratios=ratios)\
            .get_subfigures_kw()
        kw |= mpl_subfigures_kw
        out_n = kw['nrows'] * kw['ncols']
        out = self._raw.subfigures(**kw)
        
        if out_n == 1:
            out = SubFigure(out)
        else:
            out = SubFigureArray(out)

        return out
    
    def colorbar(self, mappable: ScalarMappable, cax: Axes = None, 
            ax: Axes|AxesArray|Iterable[Axes] = None,
            location: str = None, orientation: str = None,
            fraction: float = 0.1, shrink: float = 1.0, aspect: float = 20,
            pad: float = 0, ticks: list[float] = None,
            label: str = None, **mpl_colorbar_kw
        ):
        if isinstance(mappable, MplObj):
            mappable = mappable._raw
        if isinstance(cax, Axes):
            cax = cax._raw
        
        if isinstance(ax, Axes):
            ax = ax._raw
        elif isinstance(ax, AxesArray):
            ax = [_ax._raw for _ax in ax.to_list()]
        elif isinstance(ax, Iterable):
            ax = [(_ax._raw if isinstance(_ax, Axes) else _ax) for _ax in ax]
        
        art = self._raw.colorbar(mappable=mappable, cax=cax, ax=ax,
            location=location, orientation=orientation,
            fraction=fraction, shrink=shrink, aspect=aspect,
            pad=pad, ticks=ticks, label=label, **mpl_colorbar_kw)
        self._last_draw.append(art)
        
        return self

class Figure(FigureBase):
    
    Raw = mpl_figure.Figure
    
    def __init__(self, raw: Raw = None, **kw) -> None:
        super().__init__(raw=raw, **kw)
        
        self._raw : Figure.Raw
        

class SubFigure(FigureBase):
    
    Raw = mpl_figure.SubFigure
    
    def __init__(self, raw: Raw = None, **kw) -> None:
        super().__init__(raw, **kw)
        
        self._raw: SubFigure.Raw
        
class SubFigureArray(HasSimpleRepr):
    
    Raw = 'np.ndarray[Any, SubFigure.Raw]'
    
    def __init__(self, sub_figure_array: Raw = None, **kw) -> None:
        
        super().__init__(**kw)
        
        if isinstance(sub_figure_array, SubFigureArray):
            arr = sub_figure_array._sub_figure_array
        else:
            arr = np.asarray(sub_figure_array)
        shape = arr.shape
        
        arr = [ (f if isinstance(f, SubFigure) else SubFigure(f)) 
               for f in arr.flat ]
        arr = np.array(arr).reshape(shape)
            
        self._sub_figure_array = arr
        
    def subplots(self, n = 1, share = False, extent = None, space = None, 
            ratios = None, mpl_subplot_kw=None, **mpl_gridspec_kw):
        kw = dict(n=n, share=share, extent=extent, space=space, 
                  ratios=ratios, mpl_subplot_kw=mpl_subplot_kw, 
                  **mpl_gridspec_kw)
        out = np.empty(self._sub_figure_array.shape, dtype=object)
        out[...] = [f.subplots(**kw) for f in self]
        return out
    
    def __getitem__(self, key) -> Union[SubFigure, SubFigureArray]:
        val = self._sub_figure_array[key]
        if not isinstance(val, SubFigure):
            val = SubFigureArray(val)
        return val
    
    def __len__(self) -> int:
        return len(self._sub_figure_array)
    
    def __iter__(self) -> Union[Iterator[SubFigure], Iterator[SubFigureArray]]:
        for i in range(len(self)):
            yield self[i]