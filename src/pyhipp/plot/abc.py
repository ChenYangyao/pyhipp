from typing import Any, Self, Generic, TypeVar, ClassVar
import matplotlib as mpl
from matplotlib.artist import Artist
import matplotlib.pylab as plt
from matplotlib import axes as mpl_axes, figure as mpl_figure, \
    artist as mpl_artist, colors as mpl_colors, cm as mpl_cm, \
    axis as mpl_axis

Raw = TypeVar('Raw')

class MplObj(Generic[Raw]):
    def __init__(self, raw: Raw = None, **kw) -> None:
        super().__init__(**kw)    
        self._raw = raw        
        
class Artist(MplObj[mpl_artist.Artist]):
    Raw = mpl_artist.Artist
        