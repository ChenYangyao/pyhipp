from __future__ import annotations
import typing
from typing import Callable, Self
import numpy as np
from dataclasses import dataclass
from .. import abc


class Grid1d(abc.HasDictRepr):

    repr_attr_keys = ('locs', )

    def __init__(self, locs: np.ndarray | slice | Grid1d, **base_kw) -> None:
        super().__init__(**base_kw)

        if isinstance(locs, Grid1d):
            locs = locs.locs
        elif isinstance(locs, slice):
            locs = np.mgrid[locs]
        locs = np.array(locs, dtype=np.float64)
        assert locs.ndim == 1 and len(locs) >= 1 \
            and (np.diff(locs) >= 0.).all(), locs

        self.locs = locs

    @property
    def n_locs(self) -> int:
        return len(self.locs)

    @property
    def centers(self) -> Self:
        locs = self.locs
        cent_locs = 0.5*(locs[:-1] + locs[1:])
        return Grid1d(cent_locs)

    @property
    def extent(self) -> tuple[float, float]:
        locs = self.locs
        return float(locs[0]), float(locs[-1])

    def locate(self, x: np.ndarray):
        '''
        Return the bin index of x in self.locs.
        Return -1, if x < locs[0]. Return len(locs) - 1, if x >= locs[-1].
        '''
        return np.searchsorted(self.locs, x, side='right') - 1


class _GridMdAt:
    def __getitem__(self, grids_1d) -> GridMd:
        if isinstance(grids_1d, tuple):
            return GridMd(*grids_1d)
        return GridMd(grids_1d)


@dataclass
class _AppliedResult:
    values: np.ndarray
    locs: tuple[np.ndarray]
    locs_1d: tuple[np.ndarray]


class GridMd(abc.HasDictRepr):

    '''
    4 x 4 grids, spatial extents [0.0, 1.0] x [0.0, 1.0]:
    ```py
    GridMd.at[0.:1.:4j, 2.:3.:4j]
    ```
    '''

    Grid1DInitializer = np.ndarray | slice | Grid1d

    repr_attr_keys = ('n_dims', 'n_locs', 'locs_1d')

    at = _GridMdAt()

    def __init__(self, *grids_1d: Grid1DInitializer, **base_kw) -> None:

        super().__init__(**base_kw)
        grids_1d = tuple(Grid1d(g) for g in grids_1d)
        assert len(grids_1d) >= 1

        self.grids_1d = grids_1d

    @staticmethod
    def from_data_range(*data_1d, p_lo=0., p_hi=1.,
                        n: int | np.ndarray = 32) -> GridMd:
        if np.isscalar(n):
            n = np.full(len(data_1d), n)
        grids_1d = []
        for _data_1d, _n in zip(data_1d, n):
            min, max = np.quantile(_data_1d, (p_lo, p_hi))
            grids_1d.append(np.linspace(min, max, _n))
        return GridMd(*grids_1d)

    def applied(self, f: Callable, xy_indexing=True, stack_input=True):
        locs = self.locs_xy if xy_indexing else self.locs
        locs_flat = tuple(loc.ravel() for loc in locs)
        if stack_input:
            locs_flat = np.column_stack(locs_flat)
            out = f(locs_flat)
        else:
            out = f(*locs_flat)
        if isinstance(out, np.ndarray):
            out = self.__out_shaping(out, locs)
        else:
            out = [self.__out_shaping(_out, locs) for _out in out]
        return _AppliedResult(out, locs, self.locs_1d)

    @staticmethod
    def __out_shaping(out: np.ndarray, locs: np.ndarray) -> np.ndarray:
        out_shape = locs[0].shape + out.shape[1:]
        out = np.reshape(out, out_shape)
        return out

    @property
    def n_dims(self):
        return len(self.grids_1d)

    @property
    def n_locs(self) -> tuple[int, ...]:
        return tuple(g.n_locs for g in self.grids_1d)

    @property
    def locs(self) -> tuple[np.ndarray, ...]:
        return tuple(np.meshgrid(*self.locs_1d, indexing='ij'))

    @property
    def locs_xy(self) -> tuple[np.ndarray, ...]:
        return tuple(np.meshgrid(*self.locs_1d, indexing='xy'))    

    @property
    def locs_1d(self) -> tuple[np.ndarray, ...]:
        return tuple(g.locs for g in self.grids_1d)

    @property
    def centers(self) -> Self:
        return GridMd(*(g.centers for g in self.grids_1d))

    @property
    def extent(self) -> tuple[tuple[float, float], ...]:
        return tuple(g.extent for g in self.grids_1d)

    def locate(self, x: np.ndarray) -> tuple[np.ndarray, ...]:
        '''
        @x: ndarray (N, M), N M-dimensional data points.
        '''
        x = np.asarray(x).T
        return tuple(g.locate(_x) for _x, g in zip(x, self.grids_1d))
