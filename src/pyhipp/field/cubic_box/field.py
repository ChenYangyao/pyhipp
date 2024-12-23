from __future__ import annotations
import numpy as np
from pyhipp.core import abc
from typing import Union, Sequence, Tuple, Self
from numba.experimental import jitclass
import numba
from .mesh import _Mesh, Mesh


@jitclass
class _Field:

    data: numba.float64[:, :, :]
    mesh: _Mesh

    def __init__(self, data: np.ndarray, mesh: _Mesh | None = None) -> None:
        '''
        3-D scalar field.
        
        @data: referred, not copied. Shape must be (n_grids, n_grids, n_grids).
        @mesh: referred, not copied. If none, a default mesh is created.
        '''

        assert data.ndim == 3
        n0, n1, n2 = data.shape
        assert n0 == n1 == n2

        if mesh is None:
            mesh = _Mesh(n0, 1.0)
        else:
            assert mesh.n_grids == n0

        self.data = data
        self.mesh = mesh

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def n_dims(self) -> int:
        return 3

    @property
    def shape(self) -> Tuple[int, int, int]:
        n = self.mesh.n_grids
        return (n, n, n)


class Field(abc.HasDictRepr):

    def __init__(self, impl: _Field) -> None:

        self._impl = impl

    @classmethod
    def new_zeros(cls, mesh: Mesh) -> Self:
        n = mesh.n_grids
        data = np.zeros((n, n, n), dtype=np.float64)
        impl = _Field(data, mesh._impl)
        return cls(impl)

    @classmethod
    def new_by_data(cls, data: np.ndarray, mesh: Mesh) -> Self:
        impl = _Field(data, mesh._impl)
        return cls(impl)

    def copied(self) -> Self:
        return self.new_by_data(self.data.copy(), self.mesh)

    @property
    def data(self) -> np.ndarray:
        return self._impl.data

    @property
    def mesh(self) -> Mesh:
        return Mesh(self._impl.mesh)

    @property
    def dtype(self) -> np.dtype:
        return self._impl.dtype

    @property
    def n_dims(self) -> int:
        return self._impl.n_dims

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self._impl.shape
