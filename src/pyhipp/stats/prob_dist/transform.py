from __future__ import annotations
import typing
from typing import Self
from scipy.interpolate import interp1d
import numpy as np
from scipy.stats import norm
from ..random import Rng

class ProbTransToNorm:

    def __init__(self, xs: np.ndarray):
        args = np.argsort(xs)
        xs_sorted = xs[args]
        n_xs = len(xs)
        assert np.diff(xs_sorted).min() > 0.0

        cum_ps = np.arange(n_xs, dtype=float) / (n_xs-1.0)
        CDF_x = interp1d(xs_sorted, cum_ps, kind='slinear')
        invCDF_x = interp1d(cum_ps, xs_sorted, kind='slinear')

        self.CDF_x = CDF_x
        self.invCDF_x = invCDF_x

        p_norm = norm()
        self.CDF_norm = p_norm.cdf
        self.invCDF_norm = p_norm.ppf

    def forw(self, xs: np.ndarray, with_norm=True):
        ys = self.CDF_x(xs)
        if with_norm:
            ys = self.invCDF_norm(ys)
        return ys

    def back(self, ys: np.ndarray, with_norm=True):
        if with_norm:
            ys = self.CDF_norm(ys)
        xs = self.invCDF_x(ys)
        return xs


class AbundanceMatching:
    @staticmethod
    def by_sample(xs_src, xs_dst, rho=1., rng: Rng | int = 0):
        '''
        Transform a `source' sample so that the resulted values follow a 
        distribution of the `destination' sample, while maintaining some 
        degree of rank ordering specified by a correlation coefficient.
        
        @xs_src: 1-D ndarray, the source sample.
        
        @xs_dst: 1-D ndarray, the destination sample.
        
        @rho: float, the correlation coefficient. 
        Negativa value implies a negative correlation.
        
        @rng: random number generator | seed.
        
        Return: 1-D ndarray, the transformed values.
        
        Constraints:
        - The input values in either xs_src or xs_dst should be unique.
        - The sizes of xs_src and xs_dst can be different.
        
        Notes:
        This method is used by Zhang et al. 2025 (Nature, 642, 47--52; see their 
        Eq. 1)
        ''' 
        rng = Rng(rng)
        p_src, p_dst = ProbTransToNorm(xs_src), ProbTransToNorm(xs_dst)

        ys = p_src.forw(xs_src)
        eps = rng.normal(size=len(ys))
        ys = rho * ys + np.sqrt(1.0 - rho**2) * eps
        xs_pred = p_dst.back(ys)

        return xs_pred
