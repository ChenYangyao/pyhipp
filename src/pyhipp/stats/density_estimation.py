from __future__ import annotations
import typing
from typing import Tuple, Self, Sequence
import numpy as np
from ..core import DataDict
from ..core.dataproc import GridMd
from sklearn.neighbors import KDTree
from . import kernel as m_kernel

Kernel = m_kernel.KernelSpec

class _DensityEstimation:
    @staticmethod
    def _cvt_x_shape(x: np.ndarray):
        x = np.asarray(x)
        if x.ndim == 1:
            x = x[:, None]
        assert x.ndim == 2
        return x


class _Histogram(_DensityEstimation):
    def __init__(self, grid: GridMd, density=True) -> None:

        super().__init__()

        self._grid = grid
        self._density = bool(density)

    def fit(self, x: np.ndarray, y: np.ndarray = None,
            weight: np.ndarray = None) -> Self:

        grid, density = self._grid, self._density

        x = self._cvt_x_shape(x)
        assert x.shape[1] == grid.n_dims

        bins = grid.locs_1d
        h, _ = np.histogramdd(x, bins=bins, density=density, weights=weight)

        self._h = np.asarray(h, dtype=np.float64)

        return self

    def __call__(self, x: np.ndarray, out_fill=0.) -> np.ndarray:
        grid = self._grid
        x = self._cvt_x_shape(x)

        n_locs = grid.n_locs
        ids = grid.locate(x)
        n_x = len(ids[0])

        is_out = np.zeros(n_x, dtype=bool)
        for id, n_loc in zip(ids, n_locs):
            is_out |= (id < 0) | (id >= n_loc-1)
        clipped_ids = tuple(id.clip(0, n_loc-2)
                            for id, n_loc in zip(ids, n_locs))
        h = self._h[clipped_ids]
        h[is_out] = out_fill

        return DataDict({
            'ids': ids, 'clipped_ids': clipped_ids, 'h': h,
            'out_fill': out_fill, 'is_out': is_out,
        })


class _LocalKernelDensityEstimation(_DensityEstimation):

    Impl = KDTree

    def __init__(self, impl_kw={}) -> None:

        self.impl_kw = {**impl_kw}

    def fit(self, x: np.ndarray, y: np.ndarray = None,
            weight: np.ndarray = None) -> Self:

        x = self._cvt_x_shape(x)
        self._impl = _LocalKernelDensityEstimation.Impl(x, **self.impl_kw)

        if weight is None:
            w = 1.0 / len(x)
            weight = np.full(len(x), w)
        else:
            weight = np.asarray(weight)
            weight = weight / weight.sum()
        self._weight = weight
        self._n_dims = x.shape[1]

        return self

    def __call__(self, x: np.ndarray, max_dx: float = None,
                 kernel: Kernel = None,
                 impl_query_kw={}) -> np.ndarray:
        if kernel is not None:
            kernel = m_kernel.predefined.create(kernel)[1]
        w_tr = self._weight
        x = self._cvt_x_shape(x)
        if max_dx is None:
            max_dx = x.std(0).mean() / 10.

        ids, ds = self._impl.query_radius(
            x, r=max_dx, return_distance=True, **impl_query_kw)
        vol = m_kernel.Tophat._ball_volume(self._n_dims, max_dx)
        h_preds = []
        for d, id in zip(ds, ids):
            w = w_tr[id]
            if kernel is not None:
                w_k = kernel(d)
                n_w = w_k.size
                if n_w > 0:
                    w_k *= n_w / w_k.sum()
                w = w * w_k
            h_preds.append(w.sum() / vol)
        h_preds = np.array(h_preds, dtype=float)
        n_neighbors = np.array([len(id) for id in ids], dtype=int)
        neighbor_max_dx = np.array([(d.max() if d.size > 0 else 0.)
                                    for d in ds], dtype=float)
        return DataDict({
            'h': h_preds, 'n_neighbors': n_neighbors,
            'neighbor_max_dx': neighbor_max_dx,
        })


class _KnnDensityEstimation(_DensityEstimation):

    Impl = KDTree

    def __init__(self, impl_kw={}) -> None:

        self.impl_kw = {**impl_kw}

    def fit(self, x: np.ndarray, y: np.ndarray = None,
            weight: np.ndarray = None) -> Self:

        x = self._cvt_x_shape(x)
        self._impl = _LocalKernelDensityEstimation.Impl(x, **self.impl_kw)

        if weight is None:
            w = 1.0 / len(x)
            weight = np.full(len(x), w)
        else:
            weight = np.asarray(weight)
            weight = weight / weight.sum()
        self._weight = weight
        self._n_dims = x.shape[1]

        return self

    def __call__(self, x: np.ndarray, k: int = 32,
                 max_dx: float = None,
                 kernel: Kernel = None,
                 impl_query_kw={}) -> np.ndarray:

        x = self._cvt_x_shape(x)
        ds, ids = self._impl.query(x, k=k,
                                   return_distance=True, **impl_query_kw)
        r_max: np.ndarray = ds.max(1)
        w = self._weight[ids]

        if max_dx is not None:
            mask = ds < max_dx
            r_max.clip(max=max_dx, out=r_max)
        else:
            mask = np.ones_like(ds, dtype=bool)
        vol = m_kernel.Tophat._ball_volume(self._n_dims, r_max)

        n_w = mask.sum(1)
        if kernel is not None:
            w_k = m_kernel.predefined.create(kernel)[1](ds)
            w_k_sum = (w_k * mask).sum(1)
            w_k_sum[n_w == 0] = 1.0e-10
            w_k *= (n_w / w_k_sum)[:, None]
            w *= w_k

        h_preds = (w * mask).sum(1) / vol
        return DataDict({
            'h': h_preds, 'n_neighbors': n_w,
            'neighbor_max_dx': r_max,
        })


class DensityEstimationND:

    @staticmethod
    def by_histogram(
            x: np.ndarray, x_pred: np.ndarray, grid: GridMd,
            density=True, weight: np.ndarray = None, out_fill=0.) -> np.ndarray:
        '''
        Examples
        --------
        ## 1-D density estimation.
        
        ```py        
        num_pts = 10000
        x = np.random.normal(1., 3., size=num_pts)
        grid = dp.GridMd.at[-8.:8.:32j]
        gridp = grid.centers
        res = gridp.applied(lambda xp: DensityEstimationND.by_histogram(
            x, xp, grid=grid)['h'])
        xp, yp = res.locs_1d[0], res.values

        fig, ax = plot.subplots(
            1, figsize=4.5, extent=[1., 1., 0., 0.], layout='none')
        ax.plot(xp, yp)
        ```

        ## 2-D density estimation with weights.
        
        ```py
        num_pts = 1000000
        x, y = np.random.uniform(0, 10, (2, num_pts))
        z = np.abs(np.sin(x) * np.sin(y) + np.random.normal(0, 0.1, num_pts))

        X = np.column_stack([x, y])
        grid = dp.GridMd.at[0.:10.:32j, 0.:10.:32j]
        gridp = dp.GridMd.at[-1.:11.:32j, -1.:11.:32j]
        res = gridp.applied(
            lambda Xp: DensityEstimationND.by_histogram(
                X, Xp, grid=grid, weight=z)['h'])
        (xp, yp), zp = res.locs_1d, res.values

        fig, ax = plot.subplots(1, figsize=4.5,
                                extent=[1., 1., 0., 0.], layout='none')
        ax._raw.contourf(xp, yp, zp, cmap='rainbow', levels=12)
        ax._raw.contour(xp, yp, zp, colors='k', levels=12,
                        linewidths=[1, 2, 3], linestyles=['-', (0, (1,1))])
        ```        
        '''
        hist = _Histogram(grid, density).fit(x, weight=weight)
        return hist(x_pred, out_fill=out_fill)

    @staticmethod
    def by_local_kernel(
        x: np.ndarray, x_pred: np.ndarray, max_dx: float = None,
        kernel: Kernel = None, weight: np.ndarray = None,
        impl_kw={}, impl_query_kw={},
    ):
        '''
        Exaples
        -------
        
        ```py
        x = np.random.randn(50000, 2) * (1., 3.) + (0., 1.)

        grid = dp.GridMd.at[-3.:4.:32j, -7.5:11.:32j]
        kernel = kernel = ('gaussian', {'sigma': .3})
        lk = stats.DensityEstimationND.by_local_kernel
        res = grid.applied(lambda xp: lk(x, xp, max_dx=.6, kernel=kernel)['h'])
        (xp, yp), zp = res.locs_1d, res.values

        fig, ax = plot.subplots(1, figsize=4.5, extent=[1., 1., 0., 0.], 
            layout='none')

        # scatters, masked out by the outermost contours
        ax.c('k').scatter(x[:, 0], x[:, 1])
        levels = stats.Quantile('upper_3sigmas')(
            zp.ravel(), weight=zp.ravel())
        ax._raw.contourf(
            xp, yp, zp, levels=levels[[0, -1]], colors='white', extend='max')

        # meshs, masked in by the outermost contours.
        zp_ma = np.ma.array(zp, mask=zp <= levels[0])
        ax._raw.pcolormesh(xp, yp, zp_ma, cmap='Greys')

        # contours
        ax._raw.contour(xp, yp, zp, levels=levels, 
            colors='blue', linewidths=3.)

        ax.lim(*grid.extent)
        ```
        '''
        lk = _LocalKernelDensityEstimation(impl_kw).fit(x, weight=weight)
        return lk(
            x_pred, max_dx=max_dx, kernel=kernel,
            impl_query_kw=impl_query_kw)

    @staticmethod
    def by_knn(
        x: np.ndarray, x_pred: np.ndarray, k: int = 32,
        max_dx: float = None, kernel: Kernel = None,
        weight: np.ndarray = None,
        impl_kw={}, impl_query_kw={},
    ):
        '''
        For a large value of k, `kernel` itself does not solve the distortion 
        of estimated PDF. Instead, `max_dx` should be used to control the 
        sizes of local domains.
        '''
        knn = _KnnDensityEstimation(impl_kw).fit(x, weight=weight)
        return knn(
            x_pred, k=k, max_dx=max_dx, kernel=kernel,
            impl_query_kw=impl_query_kw)


class DensityEstimation1D(DensityEstimationND):
    pass
