from __future__ import annotations
import typing
from typing import Any, Tuple, ClassVar, Mapping, Iterator, Iterable
from pyhipp.core import abc, DataDict
import numpy as np


class Reduce(abc.HasName, abc.HasDictRepr):

    '''
    Examples
    --------
    
    Use reduction instances to reduce data, optionally with weights 
    and/or masks.
    
    ```py
    x, weight = np.array([1, 2, 3, 4, 5]), np.array([10., 20., 30., 40., 50.])
    mask = np.array([True, True, False, True, False])
    x_2d, weight_2d = np.array(
        [x, x*2, x*3]), np.array([weight, weight*2, weight*3])

    reductions: list[stats.Reduce] = [
        stats.Count(),
        stats.Sum(),
        stats.Mean(),
        stats.StdDev(),
        stats.Median(),
        stats.Quantile(.5),
        stats.Quantile([.25, .5, .75]),
        stats.Errorbar(),
        stats.reduction.predefined['errorbar']('mean+sd'),
    ]
    for r in reductions:
        print('reduction:', r)
        print('-- on empty array', r(x[[]]))
        print('-- on 1-D array:', r(x), r(x, weight), r(x, mask=mask))
        print('-- on 2-D array:', r(x_2d), r(x_2d, weight_2d))
    ```
    '''

    key: ClassVar[str] = None

    repr_attr_keys = ('key', 'empty_fill')

    def __init__(self, empty_fill: np.ndarray = 0, **base_kw) -> None:

        super().__init__(**base_kw)

        self.empty_fill = np.array(empty_fill)

    def __call__(self, x: np.ndarray, weight: np.ndarray = None,
                 mask: np.ndarray = None) -> np.ndarray:
        x = np.asarray(x)
        assert x.ndim == 1 or x.ndim == 2
        if weight is not None:
            weight = np.asarray(weight)
            assert weight.shape == x.shape
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            assert mask.shape == x.shape
        return self._impl(x, weight, mask)

    def _impl(self, x: np.ndarray, weight: np.ndarray = None,
              mask: np.ndarray = None) -> np.ndarray:
        return self._impl_1d(x, weight, mask) if x.ndim == 1 else \
            self._impl_2d(x, weight, mask)

    def _impl_1d(self, x: np.ndarray, weight: np.ndarray = None,
                 mask: np.ndarray = None) -> np.ndarray:
        if mask is not None:
            x = x[mask]
            if weight is not None:
                weight = weight[mask]
        if x.size == 0:
            out = self.empty_fill.copy()
        elif weight is None:
            out = self._impl_without_weight(x)
        else:
            weight = weight / weight.sum()
            out = self._impl_with_weight(x, weight)
        return out

    def _impl_without_weight(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def _impl_with_weight(
            self, x: np.ndarray, weight: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def _impl_2d(self, x: np.ndarray, weight: np.ndarray = None,
                 mask: np.ndarray = None) -> np.ndarray:
        y = []
        for i, _x in enumerate(x):
            _w = None if weight is None else weight[i]
            _m = None if mask is None else mask[i]
            y.append(self._impl_1d(_x, _w, _m))
        return np.array(y)


class Count(Reduce):
    '''
    weight is always ignored.
    '''

    key = 'count'

    def _impl(self, x: np.ndarray, weight: np.ndarray = None,
              mask: np.ndarray = None) -> np.ndarray:
        out = x.shape[-1] if mask is None else mask.sum(-1)
        return np.asarray(out)


class Sum(Reduce):
    '''
    weight is producted with x before sum.
    '''
    key = 'sum'

    def _impl(self, x: np.ndarray, weight: np.ndarray = None,
              mask: np.ndarray = None) -> np.ndarray:
        if weight is None:
            w_tot = mask
        else:
            w_tot = weight if mask is None else weight * mask

        if w_tot is not None:
            x = x * w_tot

        out = x.sum(-1)
        return np.asarray(out)


class Mean(Reduce):
    key = 'mean'

    def __init__(self, empty_fill: np.ndarray = -1.0e10, **base_kw) -> None:
        super().__init__(empty_fill, **base_kw)

    def _impl(self, x: np.ndarray, weight: np.ndarray = None,
              mask: np.ndarray = None) -> np.ndarray:
        sum, count = Sum(), Count()

        x_count = count(x, mask=mask)
        xw_sum = sum(x, weight, mask=mask)
        w_sum = x_count.copy() if weight is None else sum(weight, mask=mask)

        sel_empty = x_count < 1
        w_sum[sel_empty] = 1.
        out = np.asarray(xw_sum / w_sum)
        out[sel_empty] = self.empty_fill

        return out


class StdDev(Reduce):

    key = 'std_dev'

    def __init__(self, empty_fill: np.ndarray = -1.0e10, **base_kw) -> None:
        super().__init__(empty_fill, **base_kw)

    def _impl(self, x: np.ndarray, weight: np.ndarray = None,
              mask: np.ndarray = None) -> np.ndarray:
        mean = Mean(self.empty_fill)

        x_mean = mean(x, weight, mask)
        if x.ndim == 2:
            x_mean = x_mean[:, None]
        dx = x - x_mean
        return np.sqrt(mean(dx * dx, weight, mask))


class Quantile(Reduce):
    key = 'quantile'
    repr_attr_keys = ('ps', 'sort')
    ps_map = {
        '1sigma': (0.16, 0.84),
        '2sigma': (0.025, 0.975),
        '3sigma': (0.005, 0.995),
        'median+1sigma': (0.16, 0.5, 0.84),
        'median+2sigma': (0.025, 0.5, 0.975),
        'median+3sigma': (0.005, 0.5, 0.995),
        'lower_3sigmas': (0.68, 0.95, 0.99),
        'upper_3sigmas': (0.01, 0.05, 0.32),
    }

    def __init__(self,
                 ps: np.ndarray | str = 'median+1sigma',
                 sort=True,
                 empty_fill: np.ndarray = -1.0e10, **base_kw) -> None:

        if isinstance(ps, str):
            ps = self.ps_map[ps]
        ps = np.array(ps)

        _empty_fill = np.empty_like(ps)
        _empty_fill[...] = empty_fill

        super().__init__(_empty_fill, **base_kw)
        self.ps = ps
        self.sort = bool(sort)

    def _impl_without_weight(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(np.quantile(x, self.ps))

    def _impl_with_weight(
            self, x: np.ndarray, weight: np.ndarray) -> np.ndarray:
        if self.sort:
            idx = np.argsort(x)
            x, weight = x[idx], weight[idx]
        cdf = np.cumsum(weight)
        qs = np.interp(self.ps, cdf, x)
        return np.asarray(qs)


class Errorbar(Reduce):
    '''
    @reduce: 'mean+sd', 'median+1sigma', 'median+2sigma', 'median+3sigma'.
    '''
    key = 'errorbar'
    repr_attr_keys = ('reduce', )

    def __init__(self,
                 reduce='median+1sigma',
                 empty_fill: np.ndarray = -1.0e10,
                 **base_kw) -> None:

        super().__init__(empty_fill, **base_kw)
        self.reduce = str(reduce)

    def _impl(self, x: np.ndarray, weight: np.ndarray = None,
              mask: np.ndarray = None) -> np.ndarray:
        reduce, empty_fill = self.reduce, self.empty_fill
        if reduce == 'mean+sd':
            mean = Mean(empty_fill)(x, weight, mask)
            sd = StdDev(empty_fill)(x, weight, mask)
            out = mean, sd, sd
        else:
            out = Quantile(reduce,
                           empty_fill=empty_fill)(x, weight, mask)
            if out.ndim == 2:
                out = out.T
            x_lo, x_med, x_hi = out
            out = x_med, x_med - x_lo, x_hi - x_med

        return np.stack(out, axis=-1)


class Median(Reduce):

    key = 'median'
    repr_attr_keys = ('sort',)

    def __init__(self,
                 sort=True,
                 empty_fill: np.ndarray = -1.0e10,
                 **base_kw) -> None:
        super().__init__(empty_fill=empty_fill, **base_kw)
        self.sort = bool(sort)

    def _impl_without_weight(self, x: np.ndarray) -> np.ndarray:
        return np.median(x)

    def _impl_with_weight(
            self, x: np.ndarray, weight: np.ndarray) -> np.ndarray:
        quantile = Quantile(ps=0.5, sort=self.sort, empty_fill=self.empty_fill)
        return quantile(x, weight)


ReduceSpec = Reduce | str | tuple[str, Mapping]
ReduceSpecSubtypes = (Reduce, str, tuple)


class _Predefined(DataDict):

    def __init__(self, **base_kw) -> None:

        super().__init__(**base_kw)

        self |= {
            'count': Count,
            'cnt': Count,
            'sum': Sum,
            'mean': Mean,
            'std_dev': StdDev,
            'std': StdDev,
            'sd': StdDev,
            'quantile': Quantile,
            'qs': Quantile,
            'median': Median,
            'errorbar': Errorbar,
        }

    def create(self, reduce: ReduceSpec):
        if isinstance(reduce, Reduce):
            return reduce.key, reduce

        if isinstance(reduce, str):
            name, kw = reduce, {}
        else:
            name, kw = reduce

        return name, self[name](**kw)

    def iter(self, reduce: ReduceSpec | Iterable[ReduceSpec]) -> Iterator:
        if isinstance(reduce, ReduceSpecSubtypes):
            yield self.create(reduce)
            return

        for r in reduce:
            yield self.create(r)


predefined = _Predefined()
