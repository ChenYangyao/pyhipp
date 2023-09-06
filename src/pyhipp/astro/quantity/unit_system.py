from __future__ import annotations
from typing import Any
import astropy.units, astropy.constants
from ...core import abc

class UnitSystem(abc.HasName, abc.HasDictRepr):
    '''
    Attrs
    -----
    Astropy quantities for units, all in S.I. units:
    u_{l|t|m|v|gravity_constant|density|big_hubble}
    
    Python scalar values:
    u_v_to_kmps                        -- u_v / (km/s)
    c_gravity, c_m_sun, c_light_speed  -- G, Msun in the current unit system.
    '''
    
    # astropy modules and units
    astropy_u = astropy.units                   
    astropy_consts = astropy.constants
    
    mpc_to_m: float = astropy_u.Mpc.to('m')
    gyr_to_s: float = astropy_u.Gyr.to('s')
    msun_to_kg: float = astropy_u.Msun.to('kg')
    
    def __init__(self, 
                 u_length_in_m: float, 
                 u_time_in_s: float, 
                 u_mass_in_kg: float,
                 **kw):
        
        super().__init__(**kw)
        
        u = UnitSystem.astropy_u
        
        u_l = u_length_in_m * u.m
        u_t = u_time_in_s * u.s
        u_m = u_mass_in_kg * u.kg
        u_v = u_l / u_t
        u_gravity_constant = u_l**3 / u_t**2 / u_m
        u_density = u_m / u_l**3
        u_big_hubble = 1. / u_t
        u_v_to_kmps = self.__conv_coef(u_v, u.km / u.s)
        
        self.u_l = u_l
        self.u_t = u_t
        self.u_m = u_m
        self.u_v = u_v
        self.u_gravity_constant = u_gravity_constant
        self.u_density = u_density
        self.u_big_hubble = u_big_hubble
        
        self.u_v_to_kmps = u_v_to_kmps
        
        self.__get_const()
        
    @staticmethod
    def create_for_cosmology(hubble: float):
        '''
        Unit system of Mpc/h, Gyr/h, 1e10 Msun/h.
        '''
        U = UnitSystem
        u_l = U.mpc_to_m / hubble
        u_t = U.gyr_to_s / hubble
        u_m = U.msun_to_kg * 1.0e10 / hubble
        
        return UnitSystem(u_l, u_t, u_m)

    def to_simple_repr(self) -> dict:
        return {
            'u_l': str(self.u_l), 'u_t': str(self.u_t), 
            'u_m': str(self.u_m), 'u_v': str(self.u_v),
        }

    def __get_const(self):
        c = self.astropy_consts
        u = self.astropy_u
        
        self.c_gravity = self.__conv_coef(c.G, self.u_gravity_constant)
        self.c_m_sun = self.__conv_coef(u.Msun, self.u_m)
        self.c_light_speed = self.__conv_coef(c.c, self.u_v)
        
    @staticmethod
    def __conv_coef(x_from, x_to) -> float:
        return (x_from / x_to).to(1).value
