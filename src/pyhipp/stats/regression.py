from __future__ import annotations
from sklearn.neighbors import KDTree
from ..core import DataDict
import numpy as np
import numpy.typing as npt
from typing import Any, Union, Tuple, Mapping, Self, Iterable
import itertools
from . import reduction as m_reduction, kernel as m_kernel
from dataclasses import dataclass

SingleReduce = m_reduction.ReduceSpec
Reduce = SingleReduce | Iterable[SingleReduce]
Kernel = m_kernel.KernelSpec


class _Regression:

    @staticmethod
    def _cvt_x_shape(x: np.ndarray):
        x = np.asarray(x)
        if x.ndim == 1:
            x = x[:, None]
        assert x.ndim == 2
        return x


class _KnnRegression(_Regression):

    Impl = KDTree

    def __init__(self, impl_kw={}) -> None:
        '''
        @impl_kw: passed to Impl.
        '''
        self.impl_kw = {**impl_kw}

    def fit(self, x: np.ndarray, y: np.ndarray,
            weight: np.ndarray = None) -> Self:
        '''
        @x: array-like, shape (n_samples, n_features) or (n_samples, ).
        @y: array-like, shape (n_samples,).
        '''
        x = self._cvt_x_shape(x)
        self._impl = _KnnRegression.Impl(x, **self.impl_kw)
        self._y = np.array(y)
        self._weight = None if weight is None else np.array(weight)
        return self

    def __call__(self, x: np.ndarray, reduce: Reduce = 'mean',
                 k: int = 32,
                 max_dx: float = None,
                 kernel: Kernel = None,
                 impl_query_kw={}) -> DataDict:
        '''
        @reduce: a reduction operation or a Iterable of them.
        '''
        reduce = m_reduction.predefined.iter(reduce)

        y_tr, w_tr = self._y, self._weight

        x = self._cvt_x_shape(x)
        impl_query_kw = {'dualtree': True, **impl_query_kw}
        ds, ids = self._impl.query(
            x, k=k, return_distance=True, **impl_query_kw)

        mask = None if max_dx is None else ds < max_dx
        y = y_tr[ids]
        w = None if w_tr is None else w_tr[ids]
        if kernel is not None:
            w_k = m_kernel.predefined.create(kernel)[1](ds)
            w = w_k if w is None else w * w_k

        y_preds = []
        out = DataDict({'x': x, 'y': y_preds})
        for r_name, r_call in reduce:
            y_pred = r_call(y, weight=w, mask=mask)
            y_preds.append(y_pred)
            out[f'y_{r_name}'] = y_pred

        return out


class _LocalKernelRegression(_Regression):

    '''
    Similar to _KnnRegression, by use only radial distance to select neighbors.
    
    @impl_kw: passed to Impl.
    '''

    Impl = KDTree

    def __init__(self, impl_kw={}) -> None:
        self.impl_kw = {**impl_kw}

    def fit(self, x: np.ndarray, y: np.ndarray,
            weight: np.ndarray = None) -> Self:
        '''
        @x: array-like, shape (n_samples, n_features) or (n_samples, ).
        @y: array-like, shape (n_samples,).
        '''
        x = self._cvt_x_shape(x)
        self._impl = _LocalKernelRegression.Impl(x, **self.impl_kw)
        self._y = np.array(y)
        self._weight = None if weight is None else np.array(weight)
        return self

    def __call__(self, x: np.ndarray, reduce: Reduce = 'mean',
                 max_dx: float = None, kernel: Kernel = None,
                 impl_query_kw={}) -> DataDict:
        '''
        @reduce: a reduction operation or a Iterable of them.
        @max_dx: radius for the neighbor query. Default: std / 10.0, where 
        std is the standard deviation of x, computed along every axes and then 
        averaged.
        '''
        r_names = tuple(n for n, _ in m_reduction.predefined.iter(reduce))
        r_calls = tuple(c for _, c in m_reduction.predefined.iter(reduce))
        n_rs = len(r_names)
        if kernel is not None:
            kernel = m_kernel.predefined.create(kernel)[1]
        y_tr, w_tr = self._y, self._weight

        x = self._cvt_x_shape(x)
        if max_dx is None:
            max_dx = x.std(0).mean() / 10.0

        ids, ds = self._impl.query_radius(
            x, r=max_dx, return_distance=True, **impl_query_kw)

        y_preds = []
        for d, id in zip(ds, ids):
            y = y_tr[id]
            w = None if w_tr is None else w_tr[id]
            if kernel is not None:
                w_k = kernel(d)
                w = w_k if w is None else w * w_k
            y_pred = tuple(c(y, w) for c in r_calls)
            y_preds.append(y_pred)
        y_preds = [
            np.array([y_pred[i] for y_pred in y_preds])
            for i in range(n_rs)
        ]
        out = DataDict({'x': x, 'y': y_preds})
        for r_name, y_pred in zip(r_names, y_preds):
            out[f'y_{r_name}'] = y_pred

        return out


@dataclass
class _MaskFractionByKnnResult:
    values:     np.ndarray
    values_ma:  np.ma.MaskedArray
    invalid:    npt.NDArray[np.bool_]


class KernelRegressionND:

    @staticmethod
    def by_knn(x: np.ndarray, y: np.ndarray, x_pred: np.ndarray, k: int = 32,
               max_dx: float = None, reduce: Reduce = 'mean',
               kernel: Kernel = None, weight: np.ndarray = None,
               impl_kw={}, impl_query_kw={}) -> DataDict:
        '''
        Examples
        --------
        
        ```py
        x = np.random.uniform(0, 10, 1000)
        y = np.sin(x) + np.random.normal(0, 0.1, 1000)
        w = np.abs(y) + 1.0e-10

        k = len(x) // 10
        xp = np.linspace(0, 10, 32)
        yp, e_lo, e_hi = KernelRegression1D.by_knn(
            x, y, xp, k=k, max_dx=.3, reduce='errorbar', 
            kernel=('gaussian', {'sigma': .3}), weight=w)['y_errorbar'].T
            
        yp2 = KernelRegression1D.by_knn(
            x, y, xp, k=k, max_dx=.5, reduce='mean', 
            kernel=('gaussian', {'sigma': .02}), weight=w)['y_mean']
        ```
        '''

        knn = _KnnRegression(impl_kw=impl_kw)
        knn.fit(x, y, weight=weight)
        return knn(
            x_pred, reduce=reduce, k=k, max_dx=max_dx, kernel=kernel,
            impl_query_kw=impl_query_kw)

    @staticmethod
    def by_local_kernel(x: np.ndarray, y: np.ndarray, x_pred: np.ndarray,
                        max_dx: float = None, reduce: Reduce = 'mean',
                        kernel: Kernel = None, weight: np.ndarray = None,
                        impl_kw={}, impl_query_kw={}) -> DataDict:
        '''
        Examples
        --------
        
        ## Simple regression x -> y, with weight `w`.
        
        ```py
        x = np.random.uniform(0, 10, 1000)
        y = np.sin(x) + np.random.normal(0, 0.1, 1000)
        w = np.abs(y) + 1.0e-10

        xp = np.linspace(0, 10, 32)
        yp, e_lo, e_hi = stats.KernelRegression1D.by_local_kernel(x, y, xp, 
            max_dx=.5, reduce='errorbar', kernel=('gaussian', {'sigma': .05}),
            weight=w)['y'][0].T

        yp2 = stats.KernelRegression1D.by_local_kernel(x, y, xp, 
            max_dx=.5, reduce='mean', kernel=('gaussian', {'sigma': .05}),
            weight=w)['y_mean']

        fig, ax = plot.subplots(1, figsize=5)
        ax.c('r').errorfill(xp, yp, (e_lo, e_hi))
        ax.c('b').plot(xp, yp2)
        ```
            
        ## Auto-deal with non-sufficient neighbors.
        
        ```py
        xp = np.linspace(-5, 15, 32)
        pred, cnt = stats.KernelRegression1D.by_local_kernel(x, y, xp, 
            max_dx=.5, reduce=['errorbar', 'count'], 
            kernel=('gaussian', {'sigma': .05}),
            weight=w)['y']
        sel = cnt > 8
        xp = xp[sel]
        yp, e_lo, e_hi = pred[sel].T
        ```
        
        ## Density estimation, i.e., histogram with overlapping bins.
        
        ```py
        x = np.random.normal(0, 1, 10000)

        xp = np.linspace(-3., 3., 100)
        dx = .2
        dx_half = dx / 2.
        edges = np.linspace(-3.-dx_half, 3.+dx_half, 100+1)
        cnt = stats.KernelRegression1D.by_local_kernel(
            x, x, xp, max_dx=dx_half, reduce='count')['y'][0]
        hist = cnt / dx / len(x)

        fig, ax = plot.subplots(
            1, figsize=4.5, extent=[1., 1., 0., 0.], layout='none')
        ax.c('r').stairs(hist, edges)
        ```
        
        ## 2-D regression. Plotting the prediction on 2-D grids.
        
        ```py
        num_pts = 1000000
        x, y = np.random.uniform(0, 10, (2, num_pts))
        z = np.sin(x) * np.sin(y) + np.random.normal(0, 0.1, num_pts)
        X = np.column_stack([x, y])

        grid = dp.GridMd.at[0.:10.:32j, 0.:10.:32j]
        res = grid.applied(lambda Xp: stats.KernelRegressionND.by_local_kernel(
            X, z, Xp, max_dx=.3, kernel=('gaussian', {'sigma': .1}))['y'][0])
        (xp, yp), zp = res.locs_1d, res.values

        fig, ax = plot.subplots(1, figsize=5)
        ax._raw.contourf(xp, yp, zp, cmap='gnuplot2', levels=12)
        ax._raw.contour(xp, yp, zp, colors='k', levels=12,
                        linewidths=[1, 2,3], linestyles=['-', (0,(1,1))])
        ```
        '''

        lk = _LocalKernelRegression(impl_kw=impl_kw)
        lk.fit(x, y, weight=weight)
        return lk(
            x_pred, reduce=reduce, max_dx=max_dx, kernel=kernel,
            impl_query_kw=impl_query_kw)

    @staticmethod
    def masked_fraction_by_knn(
            x: np.ndarray, mask: np.ndarray, x_pred: np.ndarray,
            k: int = 32, max_dx: float = None, kernel: Kernel = None,
            weight: np.ndarray = None, min_count=10, impl_kw={},
            impl_query_kw={}) -> _MaskFractionByKnnResult:
        '''
        Find the fraction of masked points near individual test points `x_pred`.
        
        @x, mask, weight: training data.
        @k, max_dx, kernel, impl_query_kw: _KnnRegression query parameters.
        @impl_kw: _KnnRegression init parameters.
        @min_count: mask out the result for a test point, if less than 
            `min_count` points are found in its neighbor.
        '''

        if weight is None:
            weight = np.ones(len(x), dtype=float)
        masked_weight = mask * weight

        knn_kw = {'k': k, 'max_dx': max_dx, 'kernel': kernel,
                  'impl_query_kw': impl_query_kw}
        knn = _KnnRegression(impl_kw=impl_kw)
        w, count = knn.fit(x, weight)(x_pred, ['sum', 'count'], **knn_kw)['y']
        w_masked, = knn.fit(x, masked_weight)(x_pred, 'sum', **knn_kw)['y']

        frac = w_masked / w.clip(1.0e-10)
        invalid = count < min_count
        frac_ma = np.ma.array(frac, mask=invalid)

        return _MaskFractionByKnnResult(frac, frac_ma, invalid)


class KernelRegression1D(KernelRegressionND):
    pass
