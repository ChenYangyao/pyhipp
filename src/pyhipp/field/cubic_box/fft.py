from __future__ import annotations
import typing
from typing import Self, Tuple
import numpy as np
import scipy.fft as scipy_fft


class NdRealFFT:
    '''
    N-dimensional FFT for real input.
    
    @shape: shape of the input array. If larger than input, pad input with 
            zeros; if smaller, crop.
    @axes: target axes to transform. Default: all axes, or last len(shape) axes 
           if specified shape. A repeated axis will be transformed multiple times.
    @norm: 'forward' | 'backward' | 'ortho' | None (means 'backward').
    @overwrite_input: whether input can be overwritten as temporary space.
    @n_workers: number of workers for parallel computation. Negative for a wrap 
                around from os.cpu_count().
    '''

    def __init__(self, shape: Tuple[int, ...] = None,
                 axes: Tuple[int, ...] = None,
                 norm: str = None,
                 overwrite_input: bool = False,
                 n_workers: int = None) -> None:

        self.shape = shape
        self.axes = axes
        self.norm = norm
        self.overwrite_input = overwrite_input
        self.n_workers = n_workers

    def forward(self, x: np.ndarray) -> np.ndarray:
        kw = self.__impl_kw
        return scipy_fft.rfftn(x, **kw)

    def backward(self, x: np.ndarray) -> np.ndarray:
        kw = self.__impl_kw
        return scipy_fft.irfftn(x, **kw)

    @property
    def __impl_kw(self):
        return dict(s=self.shape, axes=self.axes, norm=self.norm,
                    overwrite_x=self.overwrite_input, workers=self.n_workers)


default_nd_real_fft = NdRealFFT()
