from __future__ import annotations
from numpy import random as npr
from typing import Union
import numpy as np
from ..core.dataproc.frame import Polar
import numba
from numba.experimental import jitclass

_spec = [
    ('_np_rng', numba.typeof(np.random.default_rng(0))),
]
_NpRng = np.random.Generator

@jitclass(_spec)
class _Rng:
    def __init__(self, np_rng: _NpRng) -> None:
        '''
        @np_rng: numpy.random.Generator.
        '''
        self._np_rng = np_rng
        
    def random(self, size=None):
        '''Uniform floating point value in [0, 1).'''
        return self._np_rng.random(size=size)
    
    def uniform(self, low=0.0, high=1.0, size=None):
        '''Uniform floating point value in [low, high).'''
        return self._np_rng.uniform(low=low, high=high, size=size)

    def standard_normal(self, size=None):
        return self._np_rng.standard_normal(size=size)

    def normal(self, loc = 0.0, scale = 1.0, size = None):
        return self._np_rng.normal(loc = loc, scale = scale, size = size)
    
    def lg_normal(self, loc = 0.0, scale = 1.0, size = None):
        '''
        Return 10.0**x, where x ~ NormalDist(loc, scale^2).
        '''
        lg_rv = self.normal(loc, scale, size)
        rv = 10.0**lg_rv
        return rv
    
    def uniform_sphere_polar(self, size=None):
        '''
        Return theta, phi.
        '''
        cos_theta = self.uniform(-1., 1., size=size)
        phi = self.uniform(0., 2.0*np.pi, size=size)   # [0, 2 pi]
        theta = np.arccos(cos_theta)                   # [0, pi]
        
        return theta, phi
        
    def uniform_sphere_cart(self, size=None):
        '''
        Return x, y, z.
        '''
        theta, phi = self.uniform_sphere_polar(size=size)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        cos_p, sin_p = np.cos(phi), np.sin(phi)
        z = cos_t
        x = sin_t * cos_p
        y = sin_t * sin_p
        return x, y, z

class Rng:
    
    Initializer = Union[None, int, npr.Generator, 'Rng']
    
    def __init__(self, seed: Initializer = 0):
        
        if isinstance(seed, npr.Generator):
            np_rng = seed
        elif isinstance(seed, Rng):
            np_rng = seed._np_rng
        else:
            assert isinstance(seed, (int, type(None)))
            np_rng = npr.default_rng(seed)
            
        self._np_rng = np_rng
    
    def random(self, size=None):
        '''Uniform floating point value in [0, 1).'''
        return self._np_rng.random(size=size)
    
    def uniform(self, low=0.0, high=1.0, size=None):
        '''Uniform floating point value in [low, high).'''
        return self._np_rng.uniform(low=low, high=high, size=size)

    def standard_normal(self, size=None):
        return self._np_rng.standard_normal(size=size)
    
    def normal(self, loc = 0.0, scale = 1.0, size = None):
        return self._np_rng.normal(loc = loc, scale = scale, size = size)
    
    def choice(self, a, size=None, replace=True, p=None, axis=0, shuffle=True):
        return self._np_rng.choice(a, size=size, replace=replace, p=p, 
            axis=axis, shuffle=shuffle)
        
    def permutation(self, a: Union[int, np.ndarray], axis: int=0) -> np.ndarray:
        '''
        Return a randomly permutated copy.
        @a: int | array-like. For int, permutate np.arange(a).
        '''
        return self._np_rng.permutation(a, axis=axis)
        
    def shuffle(self, a, axis=0):
        '''
        In-place shuffle.
        '''
        self._np_rng.shuffle(a, axis=axis)
    
    def uniform_sphere(self, size=None, stack=True, cartesian=True):
        '''
        Return theta, phi.
        '''
        cos_theta = self.uniform(-1., 1., size=size)
        phi = self.uniform(0., 2.0*np.pi, size=size)   # [0, 2 pi]
        theta = np.arccos(cos_theta)                   # [0, pi]
        
        if cartesian:
            return Polar.unit_vec_to_cart(theta, phi, stack=stack)
        
        out = theta, phi
        if stack:
            out = np.stack(out, axis=-1)
        return out
    
    def uniform_circle(self, size=None, stack=True, cartesian=True):
        theta = self.uniform(0., 2.0*np.pi, size=size)
        if not cartesian:
            return theta
        return Polar.unit_vec_to_cart_2d(theta, stack=stack)