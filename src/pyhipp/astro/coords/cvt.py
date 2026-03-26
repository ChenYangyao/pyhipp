from __future__ import annotations
import typing
from typing import Self
import numpy as np

def deg_to_rad(deg: float| np.ndarray) -> float| np.ndarray:
    return np.deg2rad(deg)

def rad_to_deg(rad: float| np.ndarray) -> float| np.ndarray:
    return np.rad2deg(rad)

def deg_to_arcsec(deg: float| np.ndarray) -> float| np.ndarray:
    return deg * 3600.

def arcsec_to_deg(arcsec: float| np.ndarray) -> float| np.ndarray:
    return arcsec / 3600.

def deg_to_arcmin(deg: float| np.ndarray) -> float| np.ndarray:
    return deg * 60.

def arcmin_to_deg(arcmin: float| np.ndarray) -> float| np.ndarray:
    return arcmin / 60.

def dms_to_deg(dms: str| tuple[float,float,float]) -> float:
    if isinstance(dms, str):
        dms = tuple(float(v) for v in dms.split(':'))
    d, m, s = dms
    deg = d + m / 60. + s / 3600.
    return deg

def hms_to_deg(hms: str| tuple[float,float,float]) -> float:
    if isinstance(hms, str):
        hms = tuple(float(v) for v in hms.split(':'))
    h, m, s = hms
    deg = (h + m / 60. + s / 3600.)*15.
    return deg


class AstroSpherical:
    '''
    This is for astronomical spherical coordinates, where the three coordinates 
    are (r, theta, phi), with r the radial distance, theta the inclination 
    (from -pi/2 to pi/2) and phi the azimuthal angle (from 0 to 2pi).
    '''
    
    @staticmethod
    def to_cartesian(r: np.ndarray, theta: np.ndarray, phi: np.ndarray):
        r, theta, phi = np.asarray(r), np.asarray(theta), np.asarray(phi)
        
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        cos_p, sin_p = np.cos(phi), np.sin(phi)
        
        x = r * cos_t * cos_p
        y = r * cos_t * sin_p
        z = r * sin_t
        
        return x, y, z
    
class PhysicalSpherical:
    '''
    This is for physical spherical coordinates, where the three coordinates are 
    (r, theta, phi), with r the radial distance, theta the polar angle (from 0 
    to pi) and phi the azimuthal angle (from 0 to 2pi).
    '''
    
    @staticmethod
    def to_cartesian(r: np.ndarray, theta: np.ndarray, phi: np.ndarray):
        r, theta, phi = np.asarray(r), np.asarray(theta), np.asarray(phi)
        
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        cos_p, sin_p = np.cos(phi), np.sin(phi)
        
        x = r * sin_t * cos_p
        y = r * sin_t * sin_p
        z = r * cos_t
        
        return x, y, z
    
    @staticmethod
    def from_cartesian(x: np.ndarray, y: np.ndarray, z: np.ndarray):
        x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
            
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x) % (2*np.pi)
        
        return r, theta, phi