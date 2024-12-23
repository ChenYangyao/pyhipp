from __future__ import annotations
import typing
from typing import Self, Callable, Dict
if typing.TYPE_CHECKING:
    from .model import LambdaCDM
from dataclasses import dataclass
from scipy.interpolate import interp1d
import h5py
import numpy as np
from functools import cached_property
from pathlib import Path
from ...core.abc import HasSimpleRepr, IsImmutable

@dataclass
class VirialProperties:
    '''
    rho: comoving.
    r: comoving.
    r_phy: physical.
    v: physical.
    '''
    m: np.ndarray
    rho: np.ndarray
    r: np.ndarray
    r_phy: np.ndarray
    v: np.ndarray

    @property
    def t_dyn(self):
        t_dyn = self.r_phy / self.v
        return t_dyn

@dataclass
class HaloMassFunction:
    '''
    lgM: log10(M_h) [10^10 Msun/h].
    dn_dlgM: dN/dlog10(M_h)/V [h^3 Mpc^{-3}].
    '''
    lgM: np.ndarray
    dn_dlgM: np.ndarray


class NFWProfileCalculator(IsImmutable):
    def __init__(self, halo_theory: HaloTheory, n_interp=256) -> None:
        
        self._halo_theory = halo_theory
        self._n_interp = n_interp
    
    @staticmethod
    def mu(x: np.ndarray|float):
        return np.log(1.0 + x) - x / (1.0 + x)
    
    @staticmethod
    def c_to_v_max2vir(c: np.ndarray|float):
        mu_at_c = NFWProfileCalculator.mu(c)
        out = np.sqrt(0.216217 * c / mu_at_c) 
        return out
    
    def v_max2vir_to_c(self, v_max2vir: np.ndarray|float):
        return self.__interp_c_and_v_max(v_max2vir)
        
    @cached_property
    def __interp_c_and_v_max(self):
        n_interp = self._n_interp
        c = np.concatenate([
            np.linspace(2.1626, 50.0, n_interp),
            np.linspace(51., 100.0, n_interp),
            np.linspace(101., 1000.0, n_interp),
        ])
        v_max2vir = self.c_to_v_max2vir(c)
        kw = {'kind': 'slinear'}
        return interp1d(v_max2vir, c, **kw)

class HaloTheory(HasSimpleRepr, IsImmutable):
    '''
    The initialization has overhead. So it is embeded into `LambdaCDM` as a 
    cached property. As a result, `LambdaCDM` is not mutable.
    '''
    VirialProperties = VirialProperties
    NFWProfileCalculator = NFWProfileCalculator

    def __init__(self, data_file: Path, model: LambdaCDM) -> None:
        super().__init__()

        self.data_file = Path(data_file)
        self.model = model

    def to_simple_repr(self) -> dict:
        return {
            'data_file': str(self.data_file),
            '__interp': self.__interp,
        }

    def rho_vir_mean(self, f: np.ndarray = 200.0,
                     z: np.ndarray = 0.0) -> np.ndarray:
        '''
        In comoving unit.
        '''
        a = 1.0 / (1.0 + z)
        rho_mean = self.model.rho_matter(z) * a**3
        return f * rho_mean

    def rho_vir_crit(self, f: np.ndarray = 200.0,
                     z: np.ndarray = 0.0) -> np.ndarray:
        '''
        In comoving unit.
        '''
        a = 1.0 / (1.0 + z)
        rho_crit = self.model.rho_crit(z) * a**3
        return f * rho_crit

    def vir_props_mean(
            self, m_vir: np.ndarray, f: np.ndarray = 200.0,
            z: np.ndarray = 0.0) -> VirialProperties:
        rho = self.rho_vir_mean(f, z)
        return self.__vir_props(m_vir, rho, z)

    def vir_props_crit(
            self, m_vir: np.ndarray, f: np.ndarray = 200.0,
            z: np.ndarray = 0.0) -> VirialProperties:
        rho = self.rho_vir_crit(f, z)
        return self.__vir_props(m_vir, rho, z)

    def r_vir(self, m_vir: np.ndarray, rho_vir: np.ndarray) -> np.ndarray:
        '''
        Expect `rho_vir` in comoving unit, and return `r_vir` also in comoving 
        unit (checked to be consistent with TNG).
        '''
        V = m_vir / rho_vir
        return (V / (4./3.*np.pi))**(1./3.)

    def v_vir(self, m_vir, r_vir, to_kmps=False) -> np.ndarray:
        '''
        Expect `r_vir` in physical unit, and return `v_vir` in physical.
        '''
        us = self.model.unit_system
        v_vir = np.sqrt(us.c_gravity * m_vir / r_vir)
        if to_kmps:
            v_vir *= us.u_v_to_kmps
        return v_vir

    def lg_sigma(self, lg_m: np.ndarray) -> np.ndarray:
        '''
        lg_m: log10(M) [10^10 Msun/h]
        '''
        return self.__interp['f_lg_sigma_at_lg_m'](lg_m)

    def dlg_sigma_dlg_m(self, lg_m: np.ndarray) -> np.ndarray:
        '''
        lg_m: log10(M) [10^10 Msun/h]
        '''
        return self.__interp['f_dlg_sigma_dlg_m_at_lg_m'](lg_m)

    def lg_delta_c(self, z: np.ndarray) -> np.ndarray:
        return self.__interp['f_lg_delta_c_at_z'](z)
    
    @cached_property
    def nfw_profile(self) -> NFWProfileCalculator:
        return NFWProfileCalculator(self)

    def mass_function(self, lgM_min, lgM_max, dlgM, z,
        mass_def = '200crit', impl_kw = {}):
        '''
        @lgM_min, lgM_max: min and mas of log10(halo mass) [10^10 Msun/h].
        '''
        if mass_def == '200crit':
            impl_kw = {
                'mdef_model': 'SOCritical',
                'mdef_params': {'overdensity': 200.0}
            } | impl_kw
        elif mass_def == '200mean':
            impl_kw = {
                'mdef_model': 'SOMean',
                'mdef_params': {'overdensity': 200.0},
            } | impl_kw
        elif mass_def == 'custom':
            pass
        else:
            raise ValueError(f'Unknown mass_def: {mass_def}')
        
        from hmf import MassFunction
        impl_kw |= {
            'Mmin': lgM_min + 10.0,                     # lg M, [Msun/h]
            'Mmax': lgM_max + 10.0 + dlgM * 0.001,
            'dlog10m': dlgM,
            'z': z,
        }
        hmf = MassFunction(**impl_kw)
        lgM, dn_dlgM = np.log10(hmf.m) - 10., hmf.dndlog10m
        
        return HaloMassFunction(lgM=lgM, dn_dlgM=dn_dlgM)

    @cached_property
    def __interp(self) -> dict[str, Callable | dict]:
        with h5py.File(str(self.data_file), 'r') as f:
            g = f['LgDeltaC']
            z, lg_delta_c = g['z'][()], g['lg_delta_c'][()]

            g = f['LgSigma']
            lg_m = g['lg_m'][()]
            lg_sigma = g['lg_sigma'][()]
            dlg_sigma_dlg_m = g['dlg_sigma_dlg_m'][()]

        kw = {'kind': 'slinear'}
        f_lg_sigma_at_lg_m = interp1d(lg_m, lg_sigma, **kw)
        f_dlg_sigma_dlg_m_at_lg_m = interp1d(lg_m, dlg_sigma_dlg_m, **kw)
        f_lg_delta_c_at_z = interp1d(z, lg_delta_c, **kw)
        return {
            'z': self.__find_quantity_info(z),
            'lg_delta_c': self.__find_quantity_info(lg_delta_c),
            'lg_m': self.__find_quantity_info(lg_m),
            'lg_sigma': self.__find_quantity_info(lg_sigma),
            'dlg_sigma_dlg_m': self.__find_quantity_info(dlg_sigma_dlg_m),

            'f_lg_sigma_at_lg_m': f_lg_sigma_at_lg_m,
            'f_dlg_sigma_dlg_m_at_lg_m': f_dlg_sigma_dlg_m_at_lg_m,
            'f_lg_delta_c_at_z': f_lg_delta_c_at_z,
        }


    def __find_quantity_info(self, x) -> Dict:
        return {
            'range': (x[0].tolist(), x[-1].tolist()),
            'step': np.diff(x).mean().tolist(),
        }

    def __vir_props(self, m: np.ndarray, rho: np.ndarray, z: np.ndarray) -> VirialProperties:
        r = self.r_vir(m, rho)
        a = 1. / (1. + z)
        r_phy = a * r
        v = self.v_vir(m, r_phy)
        return VirialProperties(m, rho, r, r_phy, v)