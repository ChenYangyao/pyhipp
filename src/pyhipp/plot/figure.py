from __future__ import annotations
from typing import Any, Union, Iterator
from .abc import MplObj, mpl_figure
from .artist_formatter import SubplotsFormatter, SubFiguresFormatter
from .axes import Axes, AxesArray
from ..core.abc import HasSimpleRepr
import numpy as np

class FigureBase(MplObj):
    
    Raw = mpl_figure.FigureBase
    
    def __init__(self, raw: Raw = None, **kw) -> None:
        
        super().__init__(raw, **kw)
        
        self._raw : FigureBase.Raw
        
    def subplots(self, n = 1, share = False, extent = None, space = None, 
            ratios = None, mpl_subplot_kw=None, **mpl_gridspec_kw):
        
        kw = SubplotsFormatter(n=n,share=share,extent=extent,space=space, 
            ratios=ratios).get_subplots_kw()
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