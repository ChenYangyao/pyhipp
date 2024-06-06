from __future__ import annotations
from .color import Color
from ..core import DataDict
import typing
import numpy as np
from collections.abc import Mapping
if typing.TYPE_CHECKING:
    from .axes import Axes

class ArtistFormatter:
    
    def __init__(self, **fmt_dict) -> None:
        self._fmt_dict = DataDict(fmt_dict)

    def __ior__(self, fmt_dict):
        d = self._fmt_dict
        for k, v in fmt_dict.items():
            if k not in d:
                raise KeyError(f'Invalid formatter key {k}')
            d[k] = v
        return self
    
    def __setitem__(self, key, value):
        d = self._fmt_dict
        if key not in d:
            raise KeyError(f'Invalid formatter key {key}')
        d[key] = value
    
    def update(self, *fmt_dicts: dict, **fmt_dict):
        for d in fmt_dicts:
            self |= d
        self |= fmt_dict
        return self
    
    def __getitem__(self, key):
        return self._fmt_dict[key]

class MarkerFormatter(ArtistFormatter):
    def __init__(self, style = 'o', s = 10,
                 ec = 'k', fc = 'k', ea = 1.0, fa = 0.3,
                 elw = 1.0) -> None:
                
        super().__init__(style=style, ec=ec, fc=fc, ea=ea, fa=fa, s=s, elw=elw)
        
    def fc(self, c = None, a = None):
        if c is not None:
            self.update(fc=c)
        if a is not None:
            self.update(fa=a)
        return self
    
    def ec(self, c = None, a = None):
        if c is not None:
            self.update(ec=c)
        if a is not None:
            self.update(ea=a)
        return self
        
    def c(self, c = None, a = None):
        self.fc(c, a)
        self.ec(c, a)

        return self

    def get_ec(self) -> Color:
        return Color(*self['ec', 'ea'])
    
    def get_fc(self) -> Color:
        return Color(*self['fc', 'fa'])

class LineFormatter(ArtistFormatter):
    def __init__(self, c = 'k', a = 1.0, 
                 lw = 2.5, ls = '-') -> None:
        
        super().__init__(c=c, a=a, lw=lw, ls=ls)
        
    def c(self, c = None, a = None):
        if c is not None:
            self.update(c=c)
        if a is not None:
            self.update(a=a)
        
        return self
    
    def get_c(self) -> Color:
        return Color(*self['c', 'a'])
        
class ErrorBarFormatter(ArtistFormatter):
    def __init__(self, c = 'k', a = 1.0, 
                 lw = 1, 
                 capsize = 4, capthick = 1) -> None:
        
        super().__init__(c=c, a=a, lw=lw, capsize=capsize, capthick=capthick)
        

    def c(self, c = None, a = None):
        if c is not None:
            self.update(c=c)
        if a is not None:
            self.update(a=a)
        
        return self

    def get_c(self) -> Color:
        return Color(*self['c', 'a'])

class FillFormatter(ArtistFormatter):
    def __init__(self, fc = 'k', ec = 'k', fa = 0.2, ea = 1, lw = 1, 
                 ls = '-') -> None:
        super().__init__(fc=fc, ec=ec, fa=fa, ea=ea, lw=lw, ls=ls)
        
        
    def fc(self, c = None, a = None):
        if c is not None:
            self.update(fc=c)
        if a is not None:
            self.update(fa=a)
        return self
    
    def ec(self, c = None, a = None):
        if c is not None:
            self.update(ec=c)
        if a is not None:
            self.update(ea=a)
        return self
        
    def c(self, c = None, a = None):
        self.fc(c, a)
        self.ec(c, a)

        return self
    
    def get_ec(self) -> Color:
        return Color(*self['ec', 'ea'])
    
    def get_fc(self) -> Color:
        return Color(*self['fc', 'fa'])
        
        
class TextFormatter(ArtistFormatter):
    def __init__(self, use_tex = True, c = 'k', a = 1.0, s = 15):
        super().__init__(use_tex=use_tex, c=c, a=a, s=s)
        
    def wrap_text(self, text: str) -> str:
        if self['use_tex']:
            text = f'${text}$'
        return text
    
    def c(self, c = None, a = None):
        if c is not None:
            self.update(c=c)
        if a is not None:
            self.update(a=a)
        
        return self
    
    def get_c(self) -> Color:
        return Color(*self['c', 'a'])
    
    
class FrameFormatter(ArtistFormatter):
    def __init__(self, lim = None, label = None, scale = None, 
            ticks = None, ticklabels = None, tick_params = None,
            label_outer=None) -> None:
        
        super().__init__(lim=lim, label=label, scale=scale, 
            ticks=ticks, ticklabels=ticklabels, tick_params=tick_params, 
            label_outer=label_outer)
        
    def apply(self, ax: Axes):
        lim = self['lim']
        if lim is not None:
            self.__apply_lim(ax, lim)
            
        label = self['label']
        if label is not None:
            self.__apply_label(ax, label)
            
        scale = self['scale']
        if scale is not None:
            self.__apply_scale(ax, scale)
            
        ticks = self['ticks']
        if ticks is not None:
            self.__apply_ticks(ax, ticks)
            
        ticklabels = self['ticklabels']
        if ticklabels is not None:
            self.__apply_ticklabels(ax, ticklabels)
            
        tick_params = self['tick_params']
        if tick_params is not None:
            self.__apply_tick_params(ax, tick_params)
            
        label_outer = self['label_outer']
        if label_outer is not None:
            if label_outer:
                ax._raw.label_outer()
            
    def __apply_lim(self, ax: Axes, lim):
        if isinstance(lim, Mapping):
            lim = dict(**lim)
            x = lim.pop('x', None)
            y = lim.pop('y', None)
            kw = lim
        else:
            x, y = lim
            kw = {}

        mpl_ax = ax._raw
        if x is not None:
            mpl_ax.set_xlim(x, **kw)
        if y is not None:
            mpl_ax.set_ylim(y, **kw)
            
    def __apply_label(self, ax: Axes, label):
        if isinstance(label, Mapping):
            label = dict(**label)
            x = label.pop('x', None)
            y = label.pop('y', None)
            kw = label
        else:
            x, y = label
            kw = {}

        mpl_ax = ax._raw
        if x is not None:
            mpl_ax.set_xlabel(x, **kw)
        if y is not None:
            mpl_ax.set_ylabel(y, **kw)
            
    def __apply_scale(self, ax: Axes, scale):
        if isinstance(scale, Mapping):
            scale = dict(**scale)
            x = scale.pop('x', None)
            y = scale.pop('y', None)
            kw = scale
        else:
            x, y = scale
            kw = {}

        mpl_ax = ax._raw
        if x is not None:
            mpl_ax.set_xscale(x, **kw)
        if y is not None:
            mpl_ax.set_yscale(y, **kw)
            
    def __apply_ticks(self, ax: Axes, ticks):
        if isinstance(ticks, Mapping):
            ticks = dict(**ticks)
            x = ticks.pop('x', None)
            y = ticks.pop('y', None)
            kw = ticks
        else:
            x, y = ticks
            kw = {}

        mpl_ax = ax._raw
        if x is not None:
            mpl_ax.set_xticks(x, **kw)
        if y is not None:
            mpl_ax.set_yticks(y, **kw)
            
    def __apply_ticklabels(self, ax: Axes, ticklabels):
        if isinstance(ticklabels, Mapping):
            ticklabels = dict(**ticklabels)
            x = ticklabels.pop('x', None)
            y = ticklabels.pop('y', None)
            kw = ticklabels
        else:
            x, y = ticklabels
            kw = {}

        mpl_ax = ax._raw
        if x is not None:
            mpl_ax.set_xticklabels(x, **kw)
        if y is not None:
            mpl_ax.set_yticklabels(y, **kw)
            
    def __apply_tick_params(self, ax: Axes, tick_params: Mapping):
        if isinstance(tick_params, Mapping):
            tick_params = dict(**tick_params)
            x = tick_params.pop('x', None)
            y = tick_params.pop('y', None)
            both = tick_params.pop('both', None)
            kw = tick_params
        
        mpl_ax = ax._raw
        if x is not None:
            mpl_ax.tick_params(axis='x', **x, **kw)
        if y is not None:
            mpl_ax.tick_params(axis='y', **y, **kw)
        if both is not None:
            mpl_ax.tick_params(axis='both', **both, **kw)
            
class SubplotsFormatter(ArtistFormatter):
    def __init__(self, *, n = 1, share = False, extent = None, 
                 space = None, ratios = None, margin = None) -> None:
        super().__init__(n=n, share=share, extent=extent, 
                         space=space, ratios=ratios, margin=margin)
        
    def get_subplots_kw(self):
        n, share, extent, space, ratios, margin = self[
            'n', 'share', 'extent', 'space', 'ratios', 'margin']
        
        if np.isscalar(n):
            n = (1, n)
        nrows, ncols = n
            
        if np.isscalar(share):
            share = (share, share)    
        sharex, sharey = share
        
        if extent is None or np.isscalar(extent):
            extent = (extent, )*4
        elif len(extent) == 2:
            e1, e2 = extent
            extent = (e2, e2, e1, e1)
        top, right, bottom, left = extent
        
        if margin is not None:
            if np.isscalar(margin):
                margin = margin, margin, margin, margin
            elif len(margin) == 2:
                m1, m2 = margin
                margin = m2, m2, m1, m1
            top, right, bottom, left = margin
            top, right = 1.0 -  top, 1.0 - right
        
        if space is None or np.isscalar(space):
            space = space, space
        wspace, hspace = space
        
        if ratios is None:
            ratios = None, None
        width_ratios, height_ratios = ratios
        
        gridspec_kw = {
            'wspace': wspace, 'hspace': hspace,
            'width_ratios': width_ratios, 'height_ratios': height_ratios,
            'top': top, 'right': right, 'bottom': bottom, 'left': left,
        }
        kw = {
            'nrows': nrows, 'ncols': ncols,
            'sharex': sharex, 'sharey': sharey,
            'gridspec_kw': gridspec_kw,
            'subplot_kw': {},
        }
        
        return kw
    
class SubFiguresFormatter(ArtistFormatter):
    def __init__(self, n = 1, space = None, ratios = None) -> None:
        super().__init__(n=n, space=space, ratios=ratios)
        
    def get_subfigures_kw(self):
        n, space, ratios = self['n', 'space', 'ratios']
        
        if np.isscalar(n):
            n = (1, n)
        nrows, ncols = n
        
        if space is None or np.isscalar(space):
            space = space, space
        wspace, hspace = space
        
        if ratios is None:
            ratios = None, None
        width_ratios, height_ratios = ratios
        
        kw = {
            'nrows': nrows, 'ncols': ncols,
            'wspace': wspace, 'hspace': hspace,
            'width_ratios': width_ratios, 'height_ratios': height_ratios,
        }
        
        return kw