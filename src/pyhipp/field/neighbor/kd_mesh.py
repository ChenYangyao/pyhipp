from __future__ import annotations
import typing
from typing import Self
from ..cubic_box.mesh import _Mesh
import numpy as np
from numba.experimental import jitclass
import numba


@jitclass
class _KdMesh3PE:

    mesh: _Mesh
    xs: numba.float64[:, :]
    ids: numba.int64[:]
    cells: numba.int64[:]

    def __init__(self, mesh: _Mesh, xs: np.ndarray, 
                 ids: np.ndarray, cells: np.ndarray):
        self.mesh = mesh
        self.xs = xs
        self.ids = ids
        assert len(ids) == mesh.total_n_grids + 1

def _kd_mesh_3pe_from_points(xs: np.ndarray, mesh: _Mesh, move_in=True):
    '''
    @xs: no bound
    '''
    pass
