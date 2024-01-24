from __future__ import annotations
import typing
from typing import Tuple
import numpy as np
from pyhipp import plot

class Axes:
    def __init__(self, ax: plot.Axes) -> None:
        self.ax = ax
        
    def add_secondary_axis_lgzp1_to_z(self, location='top',
        ticks = [0,1,2,3,4,5,7,10,14],
        label = r'$z$', hide_first_ticks = True,
        **mpl_axes_kw):
        
        ax = self.ax
        fns = (
            lambda lgzp1: 10.0**lgzp1 - 1.,
            lambda z: np.log10(1.+z)
        )
        frame_kw = {}
        if location in ('top', 'bottom'):
            sax = ax.secondary_xaxis(location, functions=fns, **mpl_axes_kw)
            axis = 'x'
        elif location in ('left', 'right'):
            sax = ax.secondary_yaxis(location, functions=fns, **mpl_axes_kw)
            axis = 'y'
        else:
            raise ValueError(f'Invalid value: {location=}')
        kw = {
            'label': {axis: label}, 'ticks': {axis: ticks}
        }
        sax.fmt_frame(**kw)
        if hide_first_ticks:
            kw = {axis: {
                'which': 'both', location: False
            }}
            ax.tick_params(**kw)