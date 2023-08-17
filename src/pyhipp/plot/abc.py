from typing import Any
import matplotlib as mpl
import matplotlib.pylab as plt
from matplotlib import axes as mpl_axes, figure as mpl_figure, \
    artist as mpl_artist, colors as mpl_colors

class MplObj:
    def __init__(self, raw: Any = None, **kw) -> None:
        super().__init__(**kw)    
        self._raw = raw
        
        
class Artist(MplObj):
    
    Raw = mpl_artist.Artist
    
    def __init__(self, raw: Raw = None, **kw) -> None:
        
        super().__init__(raw = raw, **kw)
        
