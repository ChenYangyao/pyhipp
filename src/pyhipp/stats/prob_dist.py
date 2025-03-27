from __future__ import annotations
import typing
from typing import Self
from scipy.interpolate import interp1d
import numpy as np
from scipy.stats import norm


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
