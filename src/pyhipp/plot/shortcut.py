from .figure import Figure
from .abc import plt
import numpy as np


def figure(figsize = None, dpi = None, layout='tight', **mpl_figure_kw):
    
    if np.isscalar(figsize):
        figsize = figsize, figsize
    kw = dict(figsize=figsize, dpi = dpi, layout=layout, **mpl_figure_kw)
    
    return Figure(plt.figure(**kw))

def subplots(n = 1, share = False, extent = None, space = None, ratios = None, 
    figsize = None, layout = None, subsize = 4, figure_kw = None, 
    mpl_subplot_kw = None, margin = None, **mpl_gridspec_kw):
    
    '''
    @layout: 'constrained' | 'compressed' | 'tight' | 'none' | = None. 
    Default is 'tight' if space is None, or None otherwise.
    
    @margin: 1.0 - extent, an alternative to extent.
    '''
    
    if np.isscalar(n):
        nr, nc = 1, n
    else:
        nr, nc = n
        
    if figsize is None:
        if np.isscalar(subsize):
            subsize = subsize, subsize
        w, h = subsize
        figsize = w*nc, h*nr
    
    if layout is None and space is None:
        layout = 'tight'
    kw = dict(figsize=figsize, layout=layout)
    if figure_kw is not None:
        kw |= figure_kw
    fig = figure(**kw)
    
    kw = dict(n=n, share=share, extent=extent, space=space, ratios=ratios,
              mpl_subplot_kw=mpl_subplot_kw, margin=margin, **mpl_gridspec_kw)
    ax = fig.subplots(**kw)
    
    return fig, ax
        
def subfigures(n = 1, space = None, ratios = None, 
               figsize = None, layout = None, subsize = 5, facecolor='none',
               figure_kw = None, 
               **mpl_subfigures_kw):
    if np.isscalar(n):
        nr, nc = 1, n
    else:
        nr, nc = n
    
    if figsize is None:
        if np.isscalar(subsize):
            subsize = subsize, subsize
        w, h = subsize
        figsize = w*nc, h*nr
        
    kw = dict(figsize=figsize, layout=layout)
    if figure_kw is not None:
        kw |= figure_kw
    fig = figure(**kw)
    
    kw = dict(n=n, space=space, ratios=ratios, facecolor=facecolor, 
              **mpl_subfigures_kw)
    sub_fig = fig.subfigures(**kw)
    
    return fig, sub_fig
        
def savefig(fname: str, **mpl_savefig_kw):
    plt.savefig(fname, **mpl_savefig_kw)
        
def show():
    plt.show()
    
def close():
    plt.close()