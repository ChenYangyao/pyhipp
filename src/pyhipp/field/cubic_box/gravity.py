from __future__ import annotations
import typing
from pyhipp.core import abc
from pyhipp.io import h5
from typing import Self, Iterable
from .field import _Field, _Mesh, Field
from .fft import NdRealFFT
from .mass_assignment import _LinearShapeFn, _NoneShapeFn
import numpy as np
from numba.experimental import jitclass
import numba
from .smoothing import _Gaussian
from dataclasses import dataclass


@jitclass
class _GreenFn3d:

    def __init__(self) -> None:
        '''
        Three dimensional Green's function.
        '''
        pass

    def at_ki(self, ki: np.ndarray):
        ki_sq = np.sum(ki*ki)
        if ki_sq < 1.0e-6:
            return 0.0
        return -1.0 / ki_sq


@jitclass
class _TidalTensor:

    def __init__(self) -> None:
        pass

    def at_ki(self, ki: np.ndarray, phi: float,
              axis0: int, axis1: int):
        '''
        @phi: potential at ki.
        '''
        k0, k1 = ki[axis0], ki[axis1]
        return - k0 * k1 * phi


class TidalField:

    @dataclass
    class Result:
        l_box: float
        n_grids: int
        rho_x: np.ndarray
        delta_x: np.ndarray
        delta_k: np.ndarray
        delta_sm_k: np.ndarray
        delta_sm_x: np.ndarray
        phi_k: np.ndarray
        T_x: np.ndarray
        lam: np.ndarray

        def dump(self, group: h5.Group, flag='x'):
            group.dump({
                'l_box': self.l_box,
                'n_grids': self.n_grids,
                'rho_x': self.rho_x,
                'delta_x': self.delta_x,
                'delta_k': self.delta_k,
                'delta_sm_k': self.delta_sm_k,
                'delta_sm_x': self.delta_sm_x,
                'phi_k': self.phi_k,
                'T_x': self.T_x,
                'lam': self.lam
            }, flag=flag)

    def __init__(self, n_workers=None, r_sm=1.0, correct_shape='cic') -> None:
        '''
        @r_sm: Gaussian smooth length.
        '''
        self._n_workers = n_workers
        self._r_sm = r_sm
        
        assert correct_shape in (False, 'cic')
        self._correct_shape = correct_shape

    def run(self, rho_x: Field):
        '''
        @rho_x: density field obtained by mass_assignment.DensityField, 
        i.e., assigned with CIC, without shape correction. Can be arbitrarily 
        normalized.
        '''

        rho_x, mesh = rho_x.data, rho_x.mesh._impl
        fft = NdRealFFT(n_workers=self._n_workers, norm='ortho')
        sm = _Gaussian(self._r_sm, mesh)
        S = _LinearShapeFn(mesh) if self._correct_shape == 'cic' \
            else _NoneShapeFn()

        delta_x = self._normalize(rho_x)
        delta_k = fft.forward(delta_x)
        delta_sm_k, phi_k = self._solve_grav(delta_k, S, sm)
        delta_sm_x = fft.backward(delta_sm_k)

        N = mesh.n_grids
        T_x = np.empty((N, N, N, 3, 3), dtype=np.float64)
        for i in range(3):
            for j in range(i+1):
                Tij_k = self._find_tidal_ij(phi_k, i, j, mesh)
                Tij_x = fft.backward(Tij_k)
                T_x[..., i, j] = Tij_x
                if i != j:
                    T_x[..., j, i] = Tij_x

        lam = np.linalg.eigvalsh(T_x)

        return TidalField.Result(l_box=mesh.l_box, n_grids=mesh.n_grids,
            rho_x=rho_x, delta_x=delta_x, delta_k=delta_k,
            delta_sm_k=delta_sm_k, delta_sm_x=delta_sm_x,
            phi_k=phi_k, T_x=T_x, lam=lam)

    @staticmethod
    @numba.njit
    def _normalize(rho_x: np.ndarray):
        rho_mean = rho_x.mean()
        if rho_mean < 1.0e-10:
            raise ValueError(f'Mean density {rho_mean} is too low.')

        delta_x = rho_x / rho_mean - 1.0
        return delta_x

    @staticmethod
    @numba.njit
    def _solve_grav(delta_k: np.ndarray, S: _LinearShapeFn, sm: _Gaussian):
        mesh = sm.mesh
        N = mesh.n_grids
        Nd2 = N // 2
        Nd2p1 = Nd2 + 1
        assert delta_k.shape == (N, N, Nd2p1)

        G = _GreenFn3d()

        phi_k = np.empty_like(delta_k)
        delta_sm_k = np.empty_like(delta_k)

        ki = np.empty(3, dtype=np.float64)
        for i0 in range(N):
            ki[0] = np.float64(i0 - N if i0 > Nd2 else i0)
            for i1 in range(N):
                ki[1] = np.float64(i1 - N if i1 > Nd2 else i1)
                for i2 in range(Nd2p1):
                    ki[2] = np.float64(i2)

                    w_S = 1.0 / S.shape_at_ki_nd(ki)
                    w_sm = sm.window_at_ki(ki)
                    w_G = G.at_ki(ki)

                    _delta_k = delta_k[i0, i1, i2]
                    _delta_sm_k = _delta_k * w_S * w_sm
                    _phi_k = _delta_sm_k * w_G

                    delta_sm_k[i0, i1, i2] = _delta_sm_k
                    phi_k[i0, i1, i2] = _phi_k

        return delta_sm_k, phi_k

    @staticmethod
    @numba.njit
    def _find_tidal_ij(phi_k: np.ndarray, i: int, j: int, mesh: _Mesh):

        N = mesh.n_grids
        Nd2 = N // 2
        Nd2p1 = Nd2 + 1
        assert phi_k.shape == (N, N, Nd2p1)

        T = _TidalTensor()

        Tij_k = phi_k.copy()
        ki = np.empty(3, dtype=np.float64)
        for i0 in range(N):
            ki[0] = np.float64(i0 - N if i0 > Nd2 else i0)
            for i1 in range(N):
                ki[1] = np.float64(i1 - N if i1 > Nd2 else i1)
                for i2 in range(Nd2p1):
                    ki[2] = np.float64(i2)
                    Tij_k[i0, i1, i2] = T.at_ki(ki, phi_k[i0, i1, i2], i, j)

        return Tij_k
