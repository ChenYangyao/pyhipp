from __future__ import annotations
import typing
from typing import Self
from numba.experimental import jitclass
from .mesh import _Mesh
from .field import _Field, _Mesh, Field
from .fft import NdRealFFT
from .mass_assignment import _LinearShapeFn
import numpy as np
import numba
from pyhipp.io import h5
from dataclasses import dataclass, fields


@jitclass
class _Gaussian:

    r: float
    mesh: _Mesh
    scale: float

    def __init__(self, r: float, mesh: _Mesh) -> None:
        '''
        N-dimension Gaussian smoothing.
        '''

        l_box = mesh.l_box
        scale = -0.5 * (2.0 * np.pi * r / l_box)**2

        self.r = r
        self.mesh = mesh
        self.scale = scale

    def window_at_ki(self, ki: np.ndarray):
        '''
        @ki: wave vector, e.g., (i, j, k) in 3D, where i, j, k are 
            integer indices of the grid points in fourier space.
        '''
        scale = self.scale
        ki_sq = np.sum(ki*ki)
        return np.exp(scale * ki_sq)

    def volume(self):
        return (2 * np.pi * self.r**2)**(3.0/2.0)


@jitclass
class _Tophat:

    r: float
    mesh: _Mesh
    scale: float

    def __init__(self, r: float, mesh: _Mesh) -> None:
        '''
        N-dimension tophat smoothing.
        '''

        l_box = mesh.l_box
        scale = 2.0*np.pi*r / l_box

        self.r = r
        self.mesh = mesh
        self.scale = scale

    def window_at_ki(self, ki: np.ndarray):
        '''
        @ki: wave vector, e.g., (i, j, k) in 3D, where i, j, k are 
            integer indices of the grid points in fourier space.
        '''
        kr = np.sqrt(np.sum(ki*ki)) * self.scale
        if kr < 1.0e-6:
            return 1.0

        return 3.0 * (np.sin(kr) - kr * np.cos(kr)) / kr**3

    def volume(self):
        return 4.0/3.0 * np.pi * self.r**3

class FFTSmoothing:
    
    def __init__(self, n_workers=None, r_sm=1.0, method='gaussian') -> None:
        '''
        Correction of shape function is NOT made.
        @r_sm: smooth length.
        '''
        self._n_workers = n_workers
        self._r_sm = r_sm
        self._method = method

    def run(self, field: Field):
        data, mesh = field.data, field.mesh._impl
        fft = NdRealFFT(n_workers=self._n_workers, norm='ortho')
        method = self._method
        if method == 'gaussian':
            sm = _Gaussian(self._r_sm, mesh)
        elif method == 'tophat':
            sm = _Tophat(self._r_sm, mesh)
        else:
            raise ValueError(f'Unknown method {method}.')

        data_k = fft.forward(data)
        data_sm_k = self._smooth(data_k, sm)
        data_sm = fft.backward(data_sm_k)
        return Field.new_by_data(data_sm, mesh=field.mesh)

    @staticmethod
    @numba.njit
    def _smooth(data_k: np.ndarray, sm: _Gaussian):
        mesh = sm.mesh
        N = mesh.n_grids
        Nd2 = N // 2
        Nd2p1 = Nd2 + 1
        assert data_k.shape == (N, N, Nd2p1)
        
        data_sm_k = np.empty_like(data_k)
        ki = np.empty(3, dtype=np.float64)
        for i0 in range(N):
            ki[0] = np.float64(i0 - N if i0 > Nd2 else i0)
            for i1 in range(N):
                ki[1] = np.float64(i1 - N if i1 > Nd2 else i1)
                for i2 in range(Nd2p1):
                    ki[2] = np.float64(i2)
                    w = sm.window_at_ki(ki)
                    data_sm_k[i0, i1, i2] = data_k[i0, i1, i2] * w
        return data_sm_k
    
class FourierSpaceSmoothing:

    @dataclass
    class Result:
        l_box: float
        n_grids: int
        rho_x: np.ndarray
        delta_x: np.ndarray
        delta_k: np.ndarray
        delta_sm_k: np.ndarray
        delta_sm_x: np.ndarray

        def dump(self, group: h5.Group, flag='x'):
            out = {f.name: getattr(self, f.name) for f in fields(self)}
            group.dump(out, flag=flag)

    def __init__(self, n_workers=None, r_sm=1.0, method='gaussian') -> None:
        '''
        @r_sm: smooth length.
        '''
        self._n_workers = n_workers
        self._r_sm = r_sm
        self._method = method

    def run(self, rho_x: Field):
        '''
        @rho_x: density field obtained by mass_assignment.DensityField, 
        i.e., assigned with CIC, without shape correction. Can be arbitrarily 
        normalized.
        '''

        rho_x, mesh = rho_x.data, rho_x.mesh._impl
        fft = NdRealFFT(n_workers=self._n_workers, norm='ortho')
        method = self._method
        if method == 'gaussian':
            sm = _Gaussian(self._r_sm, mesh)
        elif method == 'tophat':
            sm = _Tophat(self._r_sm, mesh)
        else:
            raise ValueError(f'Unknown method {method}.')

        delta_x = self._normalize(rho_x)
        delta_k = fft.forward(delta_x)
        delta_sm_k = self._smooth(delta_k, sm)
        delta_sm_x = fft.backward(delta_sm_k)

        return self.Result(l_box=mesh.l_box, n_grids=mesh.n_grids,
                           rho_x=rho_x, delta_x=delta_x, delta_k=delta_k,
                           delta_sm_k=delta_sm_k, delta_sm_x=delta_sm_x)

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
    def _smooth(delta_k: np.ndarray, sm: _Gaussian):
        mesh = sm.mesh
        N = mesh.n_grids
        Nd2 = N // 2
        Nd2p1 = Nd2 + 1
        assert delta_k.shape == (N, N, Nd2p1)

        S = _LinearShapeFn(mesh)

        delta_sm_k = np.empty_like(delta_k)
        ki = np.empty(3, dtype=np.float64)
        for i0 in range(N):
            ki[0] = np.float64(i0 - N if i0 > Nd2 else i0)
            for i1 in range(N):
                ki[1] = np.float64(i1 - N if i1 > Nd2 else i1)
                for i2 in range(Nd2p1):
                    ki[2] = np.float64(i2)

                    w_CIC = 1.0 / S.shape_at_ki_nd(ki)
                    w_sm = sm.window_at_ki(ki)
                    w = w_CIC * w_sm

                    delta_sm_k[i0, i1, i2] = delta_k[i0, i1, i2] * w

        return delta_sm_k
