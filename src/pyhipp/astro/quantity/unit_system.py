from __future__ import annotations
from typing import Any
import astropy.units, astropy.constants
from ...core import abc
from numba.experimental import jitclass
import numba

@jitclass
class _Constants:
    '''
    All in CGS, i.e. [m, kg, s]
    '''
    pc_to_m: numba.float64
    kpc_to_m: numba.float64
    mpc_to_m: numba.float64
    yr_to_s: numba.float64
    kyr_to_s: numba.float64
    myr_to_s: numba.float64
    gyr_to_s: numba.float64
    msun_to_kg: numba.float64

    light_speed: numba.float64
    proton_mass: numba.float64
    gravity_constant: numba.float64
    solar_mass: numba.float64
    salpeter_time: numba.float64

    def __init__(self) -> None:

        self.mpc_to_m = 3.085677581491367e+22
        self.kpc_to_m = self.mpc_to_m * 1.0e3
        self.pc_to_m = self.mpc_to_m / 1.0e6
        self.gyr_to_s = 3.15576e+16
        self.myr_to_s = self.gyr_to_s / 1.0e3
        self.kyr_to_s = self.gyr_to_s / 1.0e6
        self.yr_to_s = self.gyr_to_s / 1.0e9
        self.msun_to_kg = 1.988409870698051e+30

        self.light_speed = 299792458.0
        self.proton_mass = 1.67262192369e-27
        self.gravity_constant = 6.6743e-11
        self.solar_mass = 1.988409870698051e+30
        
        self.salpeter_time = 450.49 * (self.gyr_to_s * 1.0e-3)
        
_constants = _Constants()

@jitclass
class _UnitSystem:

    u_length: numba.float64
    u_time: numba.float64
    u_mass: numba.float64
    u_temperature: numba.float64

    u_gravity_constant: numba.float64
    u_density: numba.float64
    u_big_hubble: numba.float64
    u_velocity: numba.float64

    u_velocity_in_kmps: numba.float64

    proton_mass: numba.float64
    solar_mass: numba.float64
    
    pc: numba.float64
    
    yr: numba.float64
    salpeter_time: numba.float64
    
    light_speed: numba.float64
    gravity_constant: numba.float64
    
    def __init__(self,
                 u_length: float,
                 u_time: float,
                 u_mass: float,
                 u_temperature: float) -> None:
        '''
        Arguments should be in SI units.
        '''

        self.u_length = u_length
        self.u_time = u_time
        self.u_mass = u_mass
        self.u_temperature = u_temperature

        self.u_gravity_constant = u_length**3 / u_time**2 / u_mass
        self.u_density = u_mass / u_length**3
        self.u_big_hubble = 1.0 / u_time
        self.u_velocity = u_length / u_time

        self.u_velocity_in_kmps = self.u_velocity / 1.0e3

        cc = _Constants()
        self.proton_mass = cc.proton_mass / u_mass
        self.solar_mass = cc.solar_mass / u_mass
        self.pc = cc.pc_to_m / u_length
        self.yr = cc.yr_to_s / u_time
        self.salpeter_time = cc.salpeter_time / u_time
        self.light_speed = cc.light_speed / self.u_velocity
        self.gravity_constant = cc.gravity_constant / self.u_gravity_constant

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
        u_e = u_m * u_v**2
        u_power = u_e / u_t
        u_angular_momentum = u_l * u_m * u_v
        u_gravity_constant = u_l**3 / u_t**2 / u_m
        u_density = u_m / u_l**3
        u_big_hubble = 1. / u_t
        u_l_to_pc = self.__conv_coef(u_l, 1.0 * u.pc)
        u_t_to_yr = self.__conv_coef(u_t, 1.0 * u.yr)
        u_m_to_sol = self.__conv_coef(u_m, 1.0 * u.Msun)
        u_v_to_kmps = self.__conv_coef(u_v, u.km / u.s)
        u_e_to_erg = self.__conv_coef(u_e, 1.0 * u.erg)
        u_power_to_ergps = self.__conv_coef(u_power, u.erg / u.s)
        
        self.u_l = u_l
        self.u_t = u_t
        self.u_m = u_m
        self.u_v = u_v
        self.u_e = u_e
        self.u_power = u_power
        self.u_angular_momentum = u_angular_momentum
        self.u_gravity_constant = u_gravity_constant
        self.u_density = u_density
        self.u_big_hubble = u_big_hubble
        
        self.u_l_to_pc = u_l_to_pc
        self.u_t_to_yr = u_t_to_yr
        self.u_m_to_sol = u_m_to_sol
        self.u_v_to_kmps = u_v_to_kmps
        self.u_e_to_erg = u_e_to_erg
        self.u_power_to_ergps = u_power_to_ergps
        
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
        self.c_m_p = self.__conv_coef(c.m_p, self.u_m)
        self.c_h_planck = self.__conv_coef(c.h, self.u_angular_momentum)
        
    @staticmethod
    def __conv_coef(x_from, x_to) -> float:
        return (x_from / x_to).to(1).value
