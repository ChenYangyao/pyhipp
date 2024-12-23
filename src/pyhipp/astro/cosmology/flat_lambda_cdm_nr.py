from __future__ import annotations
from typing import Dict, Any, Self, Callable
import numpy as np
import h5py
import json
import os
from pathlib import Path
from scipy.interpolate import interp1d
import importlib_resources
from functools import cached_property
from .param import ParamList, Param
from ...core.abc import HasName, HasSimpleRepr, HasCache, IsImmutable
from ...core import dataproc as dp
import astropy.cosmology
from ..quantity import UnitSystem
from contextlib import contextmanager
from dataclasses import dataclass
from numba.experimental import jitclass
import numba
from ..quantity.unit_system import _Constants, _UnitSystem


@numba.njit
def _make_us_for_cosmology(hubble: float) -> _UnitSystem:
    cc = _Constants()
    return _UnitSystem(
        cc.mpc_to_m / hubble,
        cc.gyr_to_s / hubble,
        cc.msun_to_kg * 1.0e10 / hubble,
        1.0,
    )


@jitclass
class _FlatLambdaCDMNR:

    hubble: numba.float64
    omega_m0: numba.float64
    omega_b0: numba.float64
    n_spec: numba.float64
    sigma_8: numba.float64

    omega_l0: numba.float64
    big_hubble0: numba.float64

    us: _UnitSystem

    def __init__(self, hubble: float, omega_m0: float,
                 omega_b0: float, n_spec: float,
                 sigma_8: float) -> None:
        '''
        NR: no radiation.
        
        @hubble: small hubble, i.e., H0 / (100 km/s/Mpc).
        @omega_m0, omega_b0: matter and baryon density at z=0, in the unit
            of the critical density at z=0
        @n_spec: spectral index of the primordial power spectrum.
        @sigma_8: functuation of the linear density field within a real-space 
            8 Mpc/h tophat window at z=0.
        '''
        pass

        self.hubble = hubble
        self.omega_m0 = omega_m0
        self.omega_b0 = omega_b0
        self.n_spec = n_spec
        self.sigma_8 = sigma_8

        self.omega_l0 = 1.0 - omega_m0
        self.big_hubble0 = 0.1022712165045695

        self.us = _make_us_for_cosmology(hubble)

    def efunc_sqr(self, z: float):
        '''
        E(z)^2.
        '''
        zp1 = z + 1.0
        return self.omega_l0 + self.omega_m0 * zp1**3

    def efunc(self, z: float):
        '''
        E(z).
        '''
        return np.sqrt(self.efunc_sqr(z))

    def omega_m(self, z):
        zp1 = z + 1.0
        return self.omega_m0 * zp1**3 / self.efunc_sqr(z)

    def omega_l(self, z):
        return self.omega_l0 / self.efunc_sqr(z)

    def baryon_fraction(self):
        return self.omega_b0 / self.omega_m0

    def big_hubble(self, z):
        '''
        H(z).
        '''
        return self.big_hubble0 * self.efunc(z)

    def a2z(self, a: float):
        return 1.0 / a - 1.0

    def z2a(self, z: float):
        return 1.0 / (z + 1.0)

    def densities(self):
        return _Densities(self)


_planck_2015 = _FlatLambdaCDMNR(hubble=0.6774,
                             omega_m0=0.3089,
                             omega_b0=0.0486,
                             n_spec=0.9667,
                             sigma_8=0.8159)


@jitclass
class _Densities:

    cosm: _FlatLambdaCDMNR

    def __init__(self, cosm: _FlatLambdaCDMNR) -> None:
        self.cosm = cosm

    def crit0(self):
        cosm = self.cosm
        us = cosm.us

        H0 = cosm.big_hubble0
        G = us.gravity_constant

        return 3.0 * H0**2 / (8.0 * np.pi * G)

    def crit(self, z):
        '''
        Critical density at z (in physical scale).
        '''
        E2 = self.cosm.efunc_sqr(z)
        return self.crit0() * E2

    def mean0(self):
        '''
        Mean matter density (DM + baryon) at z=0.
        '''
        omega_m0 = self.cosm.omega_m0
        return self.crit0() * omega_m0

    def mean(self, z):
        '''
        Mean matter density (DM + baryon) at z (in physical scale).
        '''
        zp1 = z + 1.0
        return self.mean0() * zp1**3

    def r_to_m(self, r: float):
        '''
        Mean enclosed mass for a comoving radius.
        '''
        rho_mean0 = self.mean0()
        return 4.0/3.0 * np.pi * r**3 * rho_mean0

    def m_to_r(self, mass):
        '''
        Enclosed mass to mean comoving radius for a spherical region.
        '''
        rho_mean_0 = self.mean0()
        vol = mass / rho_mean_0
        r3 = vol / (4.0/3.0 * np.pi)
        return r3**(1.0/3.0)
