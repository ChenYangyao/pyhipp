from __future__ import annotations
import typing
import numpy as np
from .mesh import Mesh, _Mesh
from .field import Field, _Field
from .mass_assignment import _LinearShapeFn
from typing import Self
from functools import cached_property
from pyhipp.io import h5
from pyhipp.core import abc
from numba import njit


class FieldInterpolator(abc.HasDictRepr):
    def __init__(self, field: Field) -> None:
        '''
        @field: shall be a scalar field.
        '''
        super().__init__()

        self._data = field.data
        self._shape_fn = _LinearShapeFn(field.mesh._impl)

    @classmethod
    def new_density_field_from_file(
            cls, group: h5.Group, key='delta_x') -> Self:
        delta_x, l_box, n_grids = group.datasets[key, 'l_box', 'n_grids']
        rho_x = delta_x + 1.
        mesh = Mesh.new(n_grids, l_box)
        field = Field.new_by_data(rho_x, mesh)
        return cls(field)

    def value_at(self, xs: np.ndarray):
        '''
        @xs: np.ndarray, shape=(n, 3), where n is the number of points.
        '''
        n = len(xs)
        assert xs.shape == (n, 3)
        return self._value_at(xs, self._data, self._shape_fn)

    @staticmethod
    @njit
    def _value_at(xs: np.ndarray, data: np.ndarray, shape_fn: _LinearShapeFn):
        n = len(xs)
        out = np.empty(n, dtype=data.dtype)
        for i, x in enumerate(xs):
            i0_l, i0_r, w0_l, w0_r = shape_fn.weights_at(x[0])
            i1_l, i1_r, w1_l, w1_r = shape_fn.weights_at(x[1])
            i2_l, i2_r, w2_l, w2_r = shape_fn.weights_at(x[2])

            out[i] = data[i0_l, i1_l, i2_l] * w0_l * w1_l * w2_l \
                + data[i0_l, i1_l, i2_r] * w0_l * w1_l * w2_r \
                + data[i0_l, i1_r, i2_l] * w0_l * w1_r * w2_l \
                + data[i0_l, i1_r, i2_r] * w0_l * w1_r * w2_r \
                + data[i0_r, i1_l, i2_l] * w0_r * w1_l * w2_l \
                + data[i0_r, i1_l, i2_r] * w0_r * w1_l * w2_r \
                + data[i0_r, i1_r, i2_l] * w0_r * w1_r * w2_l \
                + data[i0_r, i1_r, i2_r] * w0_r * w1_r * w2_r
        return out


class TidalClassifier(abc.HasDictRepr):

    web_types = {
        # formal names
        'knot': 3,
        'filament': 2,
        'sheet': 1,
        'void': 0,
        # alternative names
        'k': 3,
        'f': 2,
        's': 1,
        'v': 0,
    }

    repr_attr_keys = ('mesh', 'lam_th', 'web_types',
                      'f_knot', 'f_filament', 'f_sheet', 'f_void')

    def __init__(self, lam: np.ndarray, mesh: Mesh, *, lam_th=0.0) -> None:

        lam = np.array(lam, dtype=np.float32)
        n = mesh.n_grids
        assert lam.shape == (n, n, n, 3)
        assert np.all(lam[:, :, :, 0] <= lam[:, :, :, 1])
        assert np.all(lam[:, :, :, 1] <= lam[:, :, :, 2])
        n_lams = np.count_nonzero(lam > lam_th, axis=-1)

        self.lam = lam
        self.n_lams = n_lams
        self.mesh = mesh
        self.lam_th = lam_th

    @classmethod
    def new_from_file(cls, group: h5.Group, *, lam_th=0.0) -> Self:
        n_grids, l_box, lam = group.datasets['n_grids', 'l_box', 'lam']
        mesh = Mesh.new(n_grids, l_box)
        return cls(lam, mesh, lam_th=lam_th)

    def web_type_at(self, xs: np.ndarray):
        '''
        @xs: np.ndarray, shape=(n, 3), where n is the number of points.
        '''
        xs = np.asarray(xs)
        n_xs = len(xs)
        assert xs.shape == (n_xs, 3)
        return self._web_type_at(xs, self.mesh._impl, self.n_lams)

    def lam_at(self, xs: np.ndarray):
        xs = np.asarray(xs)
        n_xs = len(xs)
        assert xs.shape == (n_xs, 3)
        return self._lam_at(xs, self.mesh._impl, self.lam)

    @staticmethod
    @njit
    def _web_type_at(xs: np.ndarray, mesh: _Mesh, n_lams: np.ndarray):
        n_xs = len(xs)
        web_types = np.empty(n_xs, dtype=np.int64)
        for i, x in enumerate(xs):
            i0, i1, i2 = mesh.x_to_xi_nd(x)
            web_types[i] = n_lams[i0, i1, i2]
        return web_types

    @staticmethod
    @njit
    def _lam_at(xs: np.ndarray, mesh: _Mesh, lam: np.ndarray):
        n_xs = len(xs)
        lam_out = np.empty((n_xs, 3), dtype=np.float32)
        for i, x in enumerate(xs):
            i0, i1, i2 = mesh.x_to_xi_nd(x)
            lam_out[i] = lam[i0, i1, i2]
        return lam_out

    def grid_points_of_web_type(self, web_type: str):
        sel = self.n_lams == self.web_types[web_type]

        mesh = self.mesh
        n, h = mesh.n_grids, mesh.l_grid

        idx_1d = np.arange(n)
        idx_3d = np.meshgrid(idx_1d, idx_1d, idx_1d, indexing='ij')
        idx_3d = [idx[sel] for idx in idx_3d]

        n_sel = idx_3d[0].size
        x = np.empty((n_sel, 3), dtype=np.float32)
        for i in range(3):
            x[:, i] = idx_3d[i] * h

        return x

    @cached_property
    def is_knot(self):
        return self.n_lams == self.web_types['knot']

    @cached_property
    def is_filament(self):
        return self.n_lams == self.web_types['filament']

    @cached_property
    def is_sheet(self):
        return self.n_lams == self.web_types['sheet']

    @cached_property
    def is_void(self):
        return self.n_lams == self.web_types['void']

    @cached_property
    def f_knot(self):
        return self.is_knot.sum() / self.n_lams.size

    @cached_property
    def f_filament(self):
        return self.is_filament.sum() / self.n_lams.size

    @cached_property
    def f_sheet(self):
        return self.is_sheet.sum() / self.n_lams.size

    @cached_property
    def f_void(self):
        return self.is_void.sum() / self.n_lams.size
