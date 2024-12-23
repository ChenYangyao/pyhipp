from __future__ import annotations
import typing
from pyhipp.core import abc
from pyhipp.io import h5
from typing import Self, Iterable
from .field import _Field, _Mesh, Field, Mesh
import numpy as np
from numba.experimental import jitclass
import numba


@jitclass
class _LinearShapeFn:

    mesh: _Mesh

    def __init__(self, mesh: _Mesh) -> None:
        '''
        Assignment algorithm using a linear shape function, 
        i.e., cloud-in-a-Cell (CIC).
        '''
        self.mesh = mesh

    def shape_at_xi(self, xi: float):
        return max(1.0 - np.abs(xi), 0.0)

    def shape_at_x(self, x: float) -> float:
        xi = x / self.mesh.l_grid
        return self.shape_at_xi(xi)
    
    def shape_at_ki(self, ki: float) -> float:
        '''
        @k: integer grid index in Fourier space.
        '''
        if np.abs(ki) < 1.0e-6:
            return 1.0
        x = np.pi * ki / self.mesh.n_grids
        sinc = np.sin(x) / x
        return sinc * sinc
    
    def shape_at_ki_nd(self, ki: np.ndarray):
        s = 1.0
        for _ki in ki:
            s *= self.shape_at_ki(_ki)
        return s

    def shape_at_k(self, k: float):
        ki = k * self.mesh.l_box / (2.0 * np.pi)
        return self.shape_at_ki(ki)

    def weights_at(self, x: float):
        l_grid, n_grids = self.mesh.l_grid, self.mesh.n_grids

        x = x / l_grid
        x_l = np.floor(x)
        w_l = 1.0 - (x - x_l)
        w_r = 1.0 - w_l

        i_l = np.int64(x_l)
        i_r = i_l + 1
        i_l, i_r = i_l % n_grids, i_r % n_grids

        return i_l, i_r, w_l, w_r


@jitclass
class _NoneShapeFn:
    def __init__(self):
        pass
    
    def shape_at_ki(self, ki: float) -> float:
        return 1.0
    
    def shape_at_ki_nd(self, ki: np.ndarray) -> float:
        return 1.0

@jitclass
class _Linear:

    data: numba.float64[:, :, :]
    shape_fn: _LinearShapeFn

    def __init__(self, field: _Field) -> None:

        self.data = field.data
        self.shape_fn = _LinearShapeFn(field.mesh)

    def add_1(self, x: np.ndarray, weight: float = 1.0):
        '''
        Add a point to the field.
        @x: 3-D position. If out-of-box, periodic condition is used.
        '''
        d, s = self.data, self.shape_fn

        i0_l, i0_r, w0_l, w0_r = s.weights_at(x[0])
        i1_l, i1_r, w1_l, w1_r = s.weights_at(x[1])
        i2_l, i2_r, w2_l, w2_r = s.weights_at(x[2])

        d[i0_l, i1_l, i2_l] += w0_l * w1_l * w2_l * weight
        d[i0_l, i1_l, i2_r] += w0_l * w1_l * w2_r * weight
        d[i0_l, i1_r, i2_l] += w0_l * w1_r * w2_l * weight
        d[i0_l, i1_r, i2_r] += w0_l * w1_r * w2_r * weight
        d[i0_r, i1_l, i2_l] += w0_r * w1_l * w2_l * weight
        d[i0_r, i1_l, i2_r] += w0_r * w1_l * w2_r * weight
        d[i0_r, i1_r, i2_l] += w0_r * w1_r * w2_l * weight
        d[i0_r, i1_r, i2_r] += w0_r * w1_r * w2_r * weight

    def add(self, xs: np.ndarray, weights: np.ndarray = None):
        '''
        Add multiple points to the field. See add_1().
        '''
        if weights is not None:
            for x, weight in zip(xs, weights):
                self.add_1(x, weight)
        else:
            for x in xs:
                self.add_1(x)


class DensityField(abc.HasLog):

    def __init__(self, l_box: float, n_grids: int) -> None:

        mesh = _Mesh(n_grids, l_box)
        data = np.zeros((n_grids, n_grids, n_grids), dtype=np.float64)
        field = _Field(data, mesh)

        ma = _Linear(field)
        self._ma = ma

    def add(self, xs: np.ndarray, weights: np.ndarray = None):

        if weights is not None:
            assert len(xs) == len(weights)
            assert weights.ndim == 1
        assert xs.ndim == 2 and xs.shape[1] == 3

        self._ma.add(xs, weights)

    def dump(self, group: h5.Group, flag='x'):

        mesh = self._ma.shape_fn.mesh
        group.dump({
            'data': self._ma.data,
            'l_box': mesh.l_box,
            'n_grids': mesh.n_grids,
        }, flag=flag)
        
    @property
    def data(self):
        return self._ma.data

    @staticmethod
    def join_dumps(
            in_groups: Iterable[h5.Group],
            out_group: h5.Group, flag='x'):

        for i, in_group in enumerate(in_groups):
            if i == 0:
                out = in_group.load()
            else:
                data = out['data']
                data += in_group.datasets['data']
        out_group.dump(out, flag=flag)


    @staticmethod
    def load(group: h5.Group):
        data, l_box, n_grids = group.datasets['data', 'l_box', 'n_grids']
        return Field.new_by_data(data, Mesh.new(n_grids, l_box))