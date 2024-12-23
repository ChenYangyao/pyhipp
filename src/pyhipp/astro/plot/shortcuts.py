from __future__ import annotations
import typing
from typing import Tuple, Callable
import numpy as np
from pyhipp import plot
from scipy.interpolate import interp1d
from matplotlib.patches import Polygon


class Axes:
    def __init__(self, ax: plot.Axes) -> None:
        self.ax = ax

    def add_secondary_axis(
            self, fn_forward: Callable, fn_backward: Callable, location='top',
            ticks=None, label=None, hide_first_ticks=True, **mpl_axes_kw):

        ax = self.ax
        fns = fn_forward, fn_backward
        if location in ('top', 'bottom'):
            call, axis = ax.secondary_xaxis, 'x'
        elif location in ('left', 'right'):
            call, axis = ax.secondary_yaxis, 'y'
        else:
            raise ValueError(f'Invalid value: {location=}')
        sax = call(location, functions=fns, **mpl_axes_kw)

        kw = {}
        if label is not None:
            kw['label'] = {axis: label}
        if ticks is not None:
            kw['ticks'] = {axis: ticks}
        sax.fmt_frame(**kw)
        if hide_first_ticks:
            kw = {axis: {
                'which': 'both', location: False
            }}
            ax.tick_params(**kw)

        return sax

    def add_secondary_axis_lgzp1_to_z(
            self, location='top', ticks=[0, 1, 2, 3, 4, 5, 7, 10, 14],
            label=r'$z$', hide_first_ticks=True, **mpl_axes_kw):

        return self.add_secondary_axis(
            lambda lgzp1: 10.0**lgzp1 - 1.,
            lambda z: np.log10(1.+z),
            location=location, ticks=ticks, label=label,
            hide_first_ticks=hide_first_ticks,
            **mpl_axes_kw)

    def add_secondary_identical_axis(
            self, location='top', ticks=None, label=None, hide_first_ticks=True,
            **mpl_axes_kw):
        def fn(x): return x, lambda x: x
        return self.add_secondary_axis(
            *fn, location=location, ticks=ticks, label=label,
            hide_first_ticks=hide_first_ticks, **mpl_axes_kw)

    def add_secondary_axis_interp(
            self, x: np.ndarray, x_new: np.ndarray,
            interp_kw={}, **kw):

        interp_kw = dict(kind='slinear', fill_value='extrapolate') | interp_kw
        fn_forward = interp1d(x, x_new, **interp_kw)
        fn_backward = interp1d(x_new, x, **interp_kw)
        return self.add_secondary_axis(fn_forward, fn_backward, **kw)

    def fill_between_xy(self, x1, y1, x2, y2, c='k', a=1., patch_kw={}):
        '''
        @x1, y1: curve 1.
        @x2, y2: curve 2.
        '''
        x, y = np.concatenate([x1, x2[::-1]]), np.concatenate([y1, y2[::-1]])
        xy = np.column_stack([x, y])
        p = Polygon(xy, color=c, alpha=a, closed=True, **patch_kw)
        self.ax._raw.add_patch(p)

        return self
