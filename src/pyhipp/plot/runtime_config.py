from __future__ import annotations
import matplotlib as mpl
from typing import Dict, Any, Self

class RuntimeConfig:
    
    ConfigDict = Dict[str, Dict[str, Any]]
    
    default_frame_color = (0.0, 0.0, 0.0, 0.8)
    default_config_dict = {
        # text
        'font': {
            'family':   'serif',
            'weight':   'normal',
            'size':     15, 
        },
        'mathtext': {
            'fontset': 'cm', 
            'default': 'rm', 
        },
        
        # figure
        'figure': {
            'dpi':      96,  
            'figsize':  (6, 6),
        },
        # subplot
        'figure.subplot': {
            'wspace':   0.25, 
            'hspace':   0.25,
        },
        
        # axes frame
        'axes': {
            'labelsize':    16, 
            'labelweight':  'normal',
            'linewidth':    1, 
            'edgecolor':    default_frame_color, 
            'edgecolor':    default_frame_color, 
            'grid':         True,
        },
        'xtick': {
            'color':        default_frame_color,
            'labelsize':    13, 
            'direction':    'in', 
            'top':          True,
        },
        'xtick.major': {'width': 1.8, 'size': 9, },
        'xtick.minor': {'width': 1, 'size': 4, 'visible': True },
        'ytick': {'color': default_frame_color,
                  'labelsize': 13, 'direction': 'in', 'right': True},
        'ytick.major': {'width': 1.8, 'size': 9, },
        'ytick.minor': {'width': 1, 'size': 4, 'visible': True },
        'grid': {
            'color': 'gray', 
            'alpha': 0.5, 
            'linestyle': (0, (1.6, 1.6)), 
            'linewidth': 0.8,
        },
        'patch': {'edgecolor': default_frame_color, },
        
        # plot artists
        'lines': {'linewidth': 2.5},
        'errorbar': {
            'capsize': 4, 
        },
        'legend': {
            'numpoints': 2, 'scatterpoints': 3, 'markerscale': 1.0, 
            'fontsize': 15, 'title_fontsize': 16,
            
            'loc': 'best',
            'handlelength': 2.0, 'handleheight': 0.5, 
            'labelspacing': 0.1, 'handletextpad': 0.5,
            'borderpad': 0.25,
            'borderaxespad': 0.5, 'columnspacing': 1.0,
            
            'shadow': False, 'labelcolor': None, 
            'frameon': False, 'framealpha': 0.8,
        }
    }
    
    stylesheets = {
        'default': default_config_dict,
        'mathtext-it': {
            'mathtext': {
                'default': 'it'
            }
        }
    }
    
    def __init__(self, config_dict: ConfigDict = None) -> None:
        self.config_dict = RuntimeConfig.default_config_dict.copy()
        self.update(config_dict)
        
    def update(self, config_dict: ConfigDict = None) -> Self:
        '''
        @config_dict: e.g., 
        {
            'axes': {'fontsize': 16, },
            'lines': {'linewidth': 2.5, },
        }
        
        This call updates the configuration dictionary managed by the current 
        instance. A following `set_global()` reflects the changes to the 
        matplotlib backend.
        '''
        if config_dict is not None:
            self.config_dict.update(config_dict)
    
        return self
    
    def set_global(self) -> Self:
        for k, detail in self.config_dict.items():
            mpl.rc(k, **detail)
        return self

    def use_stylesheets(self, *names: str, set_global=True) -> Self:
        '''
        Available names and values are defined by dictionary 
        `RuntimeConfig.stylesheets`.
        '''
        for name in names:
            self.update(RuntimeConfig.stylesheets[name])
        if set_global:
            self.set_global()
        return self
