from __future__ import annotations
import typing
from typing import Any, Dict, Tuple, Union, Callable
from pyhipp.core import abc, DataDict
import numpy as np
from dataclasses import dataclass
from sklearn.neighbors import NearestNeighbors as KNN
if typing.TYPE_CHECKING:
    from .axes import Axes

class DensityEstimator2D(abc.HasDictRepr):
    
    @dataclass
    class MeshGridRes:
        x: np.ndarray
        y: np.ndarray
        x_m: np.ndarray
        y_m: np.ndarray
        z_m: np.ndarray
    
    def __init__(self, **kw) -> None:
        super().__init__(**kw)
    
    def fit(self, x: np.ndarray, y: np.ndarray, weight: np.ndarray) \
        -> DensityEstimator2D:
        raise NotImplementedError()
    
    def on(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
    
    def on_meshgrid(self,
                 x: np.ndarray = None, 
                 y: np.ndarray = None, 
                 x_range: Tuple[float,float] = (0.,1.), 
                 y_range: Tuple[float,float] = (0.,1.), 
                 n_x: int = 10, 
                 n_y: int = 10,
                 indexing: str = 'xy') -> MeshGridRes:
        if x is None:
            x = np.linspace(*x_range, n_x)
        if y is None:
            y = np.linspace(*y_range, n_y)
        x_m, y_m = np.meshgrid(x, y, indexing=indexing)
        z_m = self.on(x_m.reshape(-1), y_m.reshape(-1)).reshape(x_m.shape)
        return self.MeshGridRes(x, y, x_m, y_m, z_m)
        
class Histogram2D(DensityEstimator2D):
    
    repr_attr_keys = 'x_bins', 'y_bins', 'x_range', 'y_range', 'out_of_range'
    
    def __init__(self, 
                 x_bins: Union[int, np.ndarray] = 10, 
                 y_bins: Union[int, np.ndarray] = 10, 
                 x_range: Tuple[float,float] = (0., 1.), 
                 y_range: Tuple[float,float] = (0., 1.),
                 out_of_range = 'zero',
                 **kw) -> None:
        
        super().__init__(**kw)
        
        assert out_of_range in ('zero', 'fixed')
        
        self.x_bins = x_bins
        self.y_bins = y_bins
        self.x_range = x_range
        self.y_range = y_range
        self.out_of_range = out_of_range
        
    def fit(self, x: np.ndarray, y: np.ndarray, weight: np.ndarray = None)\
        -> Histogram2D:
        
        z, x_e, y_e = np.histogram2d(x, y, bins=(self.x_bins, self.y_bins), 
            range=(self.x_range, self.y_range), weights=weight,
            density=True)
        x_c = (x_e[:-1] + x_e[1:]) * .5
        y_c = (y_e[:-1] + y_e[1:]) * .5

        self.x_e, self.y_e, self.z = x_e, y_e, z
        self.x_c, self.y_c = x_c, y_c
        
        return self
        
    def on(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x_e, y_e = self.x_e, self.y_e
        n_x, n_y = len(x_e) - 1, len(y_e) - 1
        x_id = np.searchsorted(self.x_e, x) - 1
        y_id = np.searchsorted(self.y_e, y) - 1
        _x_id, _y_id = np.clip(x_id, 0, n_x-1), np.clip(y_id, 0, n_y-1)
        z = self.z[_x_id, _y_id]
        if self.out_of_range == 'zero':
            sel = (x_id != _x_id) | (y_id != _y_id)
            np.putmask(z, sel, 0)
        return z
    
class KNearestNeighbor2D(DensityEstimator2D):
    def __init__(self, k = 10, **kw) -> None:
        
        super().__init__(**kw)
        
        self.k = k
        
    def fit(self, x: np.ndarray, y: np.ndarray, weight: np.ndarray=None) \
        -> KNearestNeighbor2D:
        
        n_pts = len(x)
        k = min(self.k, n_pts)     
        knn = KNN(n_neighbors=k, algorithm='kd_tree')
        knn.fit(np.stack((x, y), axis=-1))
        
        self.k_used = k
        self.knn = knn
        self.weight = np.asarray(weight) if weight is not None else None 
        
        return self

    def on(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        k, w = self.k_used, self.weight
        r, ids = self.knn.kneighbors(np.stack((x, y), axis=-1),)
        r = r[:, -1]
        s = np.pi * r * r
        if w is None:
            z = k / s
        else:
            z = w[ids,].sum(1) / s
        return z
    
    
    
class Scatter2D(abc.HasDictRepr, abc.HasLog):
    def __init__(self, ax: Axes, x: np.ndarray, y: np.ndarray, 
        range: Tuple[Tuple[float,float], Tuple[float,float]] = None,
        n_bins: Union[int, Tuple[int,int]] = 10,
        weight: np.ndarray = None,
        z_transform: Callable = None,
        density_estimator: DensityEstimator2D = None,
        **kw
    ) -> None:
        
        super().__init__(**kw)
        
        x, y = np.asarray(x), np.asarray(y)
        if range is None:
            range = (x.min(), x.max()),(y.min(), y.max())
        if np.isscalar(n_bins):
            n_bins = n_bins, n_bins
        x_r, y_r = range
        x_b, y_b = n_bins
        x_e, y_e = np.linspace(*x_r, x_b+1), np.linspace(*y_r, y_b+1)
        x_c, y_c = .5*(x_e[1:]+x_e[:-1]), .5*(y_e[1:]+y_e[:-1])
        dx, dy = x_e[1] - x_e[0], y_e[1] - y_e[0]
        if density_estimator is None:
            density_estimator = Histogram2D(
                x_bins=x_b, y_bins=y_b, x_range=x_r, y_range=y_r
            )
        density_estimator.fit(x, y, weight=weight)

        self.data = x, y
        self.ax = ax
        self.range = range
        self.n_bins = n_bins
        self.centers = x_c, y_c
        self.edges = x_e, y_e
        self.step = dx, dy
        self.z_transform = z_transform
        self.density_estimator = density_estimator
        self.plot_outs = []
    
    def mesh(self, cmap='jet', vmin=None, vmax=None, norm=None, alpha=None,
             edgecolors='none', **pcolor_mesh_kw) -> Scatter2D:
        
        x_e, y_e = self.edges
        z = self.transformed_density_on_mesh
        kw = dict(cmap=cmap, vmin=vmin, vmax=vmax, norm=norm, alpha=alpha,
            edgecolors=edgecolors)
        kw |= pcolor_mesh_kw
        o = self.ax._raw.pcolormesh(x_e, y_e, z, **kw)
        self.plot_outs.append(o)
        return self
    
    def contour(self, ps: np.ndarray = None, use_fill = False,
                colors='k', 
                cmap=None, vmin=None, vmax=None, norm=None,
                lw=None, ls=None, **contour_kw) -> Scatter2D:
        if ps is None:
            ps = (0.99, 0.95, 0.68)
        z_levels = self.z_level_at_percentile(ps)
        
        n_ps = len(ps)
        if lw is None:
            lw = np.linspace(1, 3, n_ps)
        if ls is None:
            segs = np.linspace(3, 6, n_ps-1)
            ls = [(0, (seg, 3)) for seg in segs] + ['-']
        
        x_c, y_c = self.centers
        z = self.transformed_density_on_mesh
        kw = dict(levels=z_levels, colors=colors, cmap=cmap, vmin=vmin, 
                  vmax=vmax, norm=norm, linewidths=lw, linestyles=ls)
        kw |= contour_kw
        if use_fill:
            fn = self.ax._raw.contourf
            del kw['linewidths']
        else:
            fn = self.ax._raw.contour
        o = fn(x_c, y_c, z, **kw)
        self.plot_outs.append(o)
        
        return self
    
    def scatter(self, hidden_in_p: float = None, marker='o', s=10,
                c='k', edgecolors='none', alpha=None, linewidths=1,
                **scatter_kw) -> Scatter2D:
        x, y = self.data
        if hidden_in_p is not None:
            z = self.density_estimator.on(x, y)
            t = self.z_transform
            if t is not None:
                z = t(z)
            z_level = self.z_level_at_percentile(
                [hidden_in_p], id_shift=-1)[0]
            sel = z < z_level
            x, y = x[sel], y[sel]
            self.log(f'Select {sel.sum()}/{sel.size} points above '
                     f'z={z_level}')
        kw = dict(marker=marker, s=s, c=c, 
                  edgecolors=edgecolors, alpha=alpha, linewidths=linewidths)
        kw |= scatter_kw
        o = self.ax._raw.scatter(x, y, **kw)
        self.plot_outs.append(o)
        return self
    
    @property
    def density_on_mesh(self):
        x_c, y_c = self.centers
        return self.density_estimator.on_meshgrid(x_c, y_c)
        
    @property
    def transformed_density_on_mesh(self):
        z = self.density_on_mesh.z_m
        t = self.z_transform
        if t is not None:
            z = t(z)
        return z
    
    @property
    def cum_prob_mass(self):
        z = self.transformed_density_on_mesh
        z_sorted = np.sort(z.ravel())
        z_cum = np.cumsum(z_sorted)
        z_cum /= z_cum[-1]
        return z_sorted, z_cum
    
    def z_level_at_percentile(self, ps, id_shift=0):
        ps_lo = 1.0 - np.asarray(ps)
        z_sorted, prob = self.cum_prob_mass
        id = np.searchsorted(prob, ps_lo) + id_shift
        np.clip(id, 0, len(prob)-1, out=id)
        z_levels = z_sorted[id]
        return z_levels
        