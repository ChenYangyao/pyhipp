from __future__ import annotations
import typing
from typing import Self
import numpy as np
from numba.experimental import jitclass
import numba


class _QueryPoints:
    xs: np.ndarray
    rs: np.ndarray

    def on(self, i: int, x_ngbs: np.ndarray):
        pass


@jitclass
class _PE3Mesh:
    l_box: float
    l_grid: float
    n_grids: int

    def __init__(self, l_box: float, n_grids: int):
        '''
        Regular mesh for a periodic, equal-side (i.e. cubic) box in 3D.
        '''
        self.l_box = l_box
        self.l_grid = l_box / n_grids
        self.n_grids = n_grids

    def x2xi_1(self, x: float):
        return np.int64(np.floor(x / self.l_grid))

    def x2xi_3(self, x: np.ndarray):
        return np.floor(x / self.l_grid).astype(np.int64)

    def x2xi_f(self, x: np.ndarray):
        n = self.n_grids
        xi0, xi1, xi2 = self.x2xi_3(x) % n
        return (xi0 * n + xi1)*n + xi2

    def argsort(self, xs):
        xi_fs = np.empty(len(xs), dtype=np.int64)
        for i, x in enumerate(xs):
            xi_f = self.x2xi_f(x)
            xi_fs[i] = xi_f
        return xi_fs, np.argsort(xi_fs)


@jitclass
class _PE3:
    mesh: _PE3Mesh
    xs: numba.float64[:, :]
    cell_firsts: numba.int64[:]

    def __init__(self, mesh: _PE3Mesh,
                 xs: np.ndarray,
                 cell_firsts: np.ndarray):
        assert len(cell_firsts) == mesh.n_grids**3 + 1
        self.mesh = mesh
        self.xs = xs
        self.cell_firsts = cell_firsts

    def query_points(self, q: _QueryPoints):
        '''
        Find the neighbor points x_ngbs for each point q.xs[i] with a distance 
        q.rs[i], and call q(i, x_ngbs). 
        
        There can be multiple calls on a given i.
        
        All points within the distance q.rs[i] are guaranteed to be included,
        but some points outside the distance may also be included.
        
        This call takes a batch of points to query. Overhead on memory access
        is reduced by sorting the input points in the mesh order.
        '''
        x_dsts, r_dsts = q.xs, q.rs
        _, i_dsts = self.mesh.argsort(x_dsts)
        for i_dst in i_dsts:
            x_dst, r_dst = x_dsts[i_dst], r_dsts[i_dst]
            self.__query_points_1(i_dst, x_dst, r_dst, q)

    def __query_points_1(self, i_dst, x_dst, r_dst, q: _QueryPoints):
        mesh = self.mesh
        n = mesh.n_grids
        cell_firsts, x_ngbs = self.cell_firsts, self.xs

        lb0, lb1, lb2 = mesh.x2xi_3(x_dst - r_dst)
        ub0, ub1, ub2 = mesh.x2xi_3(x_dst + r_dst) + 1
        for i0 in range(lb0, ub0):
            i0_p = i0 % n
            for i1 in range(lb1, ub1):
                i1_p = i1 % n
                for i2 in range(lb2, ub2):
                    i2_p = i2 % n
                    i_f = (i0_p * n + i1_p) * n + i2_p
                    b, e = cell_firsts[i_f], cell_firsts[i_f+1]
                    q.on(i_dst, x_ngbs[b:e])


@numba.njit
def _PE3_from_meshing_points(mesh: _PE3Mesh, xs: np.ndarray):
    xi_fs, args = mesh.argsort(xs)
    xi_fs, xs = xi_fs[args], xs[args]
    n_xs = len(xs)

    n_tot = mesh.n_grids**3
    cell_firsts = np.zeros(n_tot+1, dtype=np.int64)
    b = 0
    for i_f in range(n_tot):
        e = b
        while e < n_xs:
            if xi_fs[e] > i_f:
                break
            e += 1
        cell_firsts[i_f+1] = e
        b = e
    return _PE3(mesh, xs, cell_firsts)
