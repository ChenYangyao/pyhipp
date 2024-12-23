from __future__ import annotations
import typing
from typing import Self
from pyhipp.core.abc import HasDictRepr
from scipy.spatial import KDTree
from ..cubic_box.mesh import Mesh
import numpy as np


class PE(HasDictRepr):
    '''
    Periodic, with equal side-length.
    @xs: must be within [0, l_box), otherwise raise an error.
    '''
    
    repr_attr_keys = ('l_box', 'n_workers')

    def __init__(self, xs: np.ndarray, l_box: float,
                 copy_data=True, balanced_tree=True, compact_nodes=True,
                 n_workers=1):

        impl = KDTree(
            xs, compact_nodes=compact_nodes, copy_data=copy_data,
            balanced_tree=balanced_tree, boxsize=l_box)

        self.impl = impl
        self.l_box = l_box
        self.n_workers = n_workers

    def query_r_1(
            self, x: np.ndarray, r: float, return_d=False) -> np.ndarray | tuple[
            np.ndarray, np.ndarray]:
        '''
        Return indices. If return_d is True, return (indices, distances).
        '''
        x = np.asarray(x)
        assert (x >= 0.0).all() and (x < self.l_box).all()

        ids = self.impl.query_ball_point(x, r, workers=self.n_workers)
        ids = np.asarray(ids, dtype=np.int64)
        if not return_d:
            return ids
        d = self.__d(x, self.impl.data[ids])
        return ids, d

    def __d(self, x1: np.ndarray, x2: np.ndarray):
        l_box = self.l_box
        l_half = 0.5 * l_box
        dx = x2 - x1
        dx[dx > l_half] -= l_box
        dx[dx < -l_half] += l_box
        return np.linalg.norm(dx, axis=-1)
