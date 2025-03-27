from __future__ import annotations
from typing import Self
from numba import njit
import numpy as np
import numba
import numbers
from numba.experimental import jitclass


@njit
def bisearch(x_sorted: np.ndarray, x_dst: numbers.Real) -> int:
    '''
    Return idx in [-1, len(x_sorted)), 
    so that x_sorted[idx] <= x_dst < x_sorted[idx+1].
    
    Requirements:
    - x_sorted is sorted in ascending order.
    - len(x_sorted) > 0.
    '''
    n_all = len(x_sorted)
    assert n_all > 0
    if x_dst < x_sorted[0]:
        return -1
    l, r = 0, n_all
    while r - l > 1:
        c = (l + r) // 2
        if x_dst < x_sorted[c]:
            r = c
        else:
            l = c
    return l


@njit
def bisearch_nearest(x_sorted: np.ndarray, x_dst: numbers.Real) -> int:
    '''
    Return idx in [0, len(x_sorted)), so that x_sorted[idx] is the nearest 
    to x_dst among all in x_sorted.
    
    Requirements: the same as bisearch(x_sorted, x_dst).
    '''
    l = bisearch(x_sorted, x_dst)
    r = l + 1
    n = len(x_sorted)
    if l < 0:
        return 0
    if r > n-1:
        return n-1
    dx_l = x_dst - x_sorted[l]
    dx_r = x_sorted[r] - x_dst
    return l if dx_l < dx_r else r


@njit
def bisearch_nearest_array(x_sorted: np.ndarray, x_dst: np.ndarray) -> int:
    '''
    Vectorized version of bisearch_nearest().
    '''
    idx = np.empty(x_dst.size, dtype=np.int64)
    for i, _x_dst in enumerate(x_dst):
        idx[i] = bisearch_nearest(x_sorted, _x_dst)
    return idx


@njit
def linear_interp_weight(x: float, x_left: float, x_right: float) \
        -> tuple[float, float]:
    '''
    Must satisfy:
    x_left <= x <= x_right
    x_left < x_right
    '''
    dx_l = x - x_left
    dx_r = x_right - x
    dx = dx_l + dx_r
    w_l, w_r = dx_r / dx, dx_l / dx
    return w_l, w_r


@njit
def bisearch_interp_weight(
    x_sorted: np.ndarray, x_dst: numbers.Real
) -> tuple[int, int, float, float]:

    l = bisearch(x_sorted, x_dst)
    if l < 0:
        return 0, 0, 0.0, 1.0
    r = l + 1
    n = len(x_sorted)
    if r > n-1:
        return n-1, n-1, 1.0, 0.0
    w_l, w_r = linear_interp_weight(x_dst, x_sorted[l], x_sorted[r])

    return l, r, w_l, w_r


@njit
def bisearch_interp(x_sorted: np.ndarray, y: np.ndarray, x_dst: numbers.Real):
    l, r, w_l, w_r = bisearch_interp_weight(x_sorted, x_dst)
    return y[l] * w_l + y[r] * w_r

@njit
def bisearch_interp_fix_oob(x_sorted: np.ndarray, y: np.ndarray, 
                            x_dst: numbers.Real, 
                            y_left: numbers.Real, y_right: numbers.Real):
    if x_dst < x_sorted[0]:
        return y_left
    if x_dst > x_sorted[-1]:
        return y_right
    return bisearch_interp(x_sorted, y, x_dst)

@njit
def bisearch_interp_2d(x1_sorted: np.ndarray, x2_sorted: np.ndarray, 
    y: np.ndarray, x1_dst: numbers.Real, x2_dst: numbers.Real):
    '''
    @y: shaped (n1, n2, ...). Interpolation is made along the first two axes.
    '''
    l1, r1, w1_l, w1_r = bisearch_interp_weight(x1_sorted, x1_dst)
    l2, r2, w2_l, w2_r = bisearch_interp_weight(x2_sorted, x2_dst)
    y_ll, y_lr, y_rl, y_rr = y[l1, l2], y[l1, r2], y[r1, l2], y[r1, r2]
    w_ll, w_lr, w_rl, w_rr = w1_l * w2_l, w1_l * w2_r, w1_r * w2_l, w1_r * w2_r
    return y_ll * w_ll + y_lr * w_lr + y_rl * w_rl + y_rr * w_rr

@jitclass
class Linear:

    _x_nodes: numba.float64[:]
    _y_nodes: numba.float64[:]

    def __init__(self, x_nodes: np.ndarray, y_nodes: np.ndarray,
                 check=True) -> None:
        '''
        @x_nodes, y_nodes: 1D arrays. 
            x_nodes must be sorted in ascending order.
            Arrays are referred, not copied.
        '''

        if check:
            n = len(x_nodes)
            assert x_nodes.shape == (n,)
            assert y_nodes.shape == (n,)
            assert (x_nodes[1:] - x_nodes[:-1]).all()
        assert len(x_nodes) >= 2

        self._x_nodes = x_nodes
        self._y_nodes = y_nodes

    def value_at(self, x: float):
        '''
        Return predicted value. Bound values are y_nodes[0], y_nodes[-1].
        '''
        return bisearch_interp(self._x_nodes, self._y_nodes, x)

    def derivative(self) -> Self:
        xs, ys = self._x_nodes, self._y_nodes
        yds = np.empty(len(ys), dtype=np.float64)

        n = len(xs)
        for i in range(n):
            i_l, i_r = i-1, i+1
            if i_l == -1:
                i_l, i_r = 0, 1
            elif i_r == n:
                i_l, i_r = n-2, n-1
            x_l, x_r = xs[i_l], xs[i_r]
            y_l, y_r = ys[i_l], ys[i_r]
            yd = (y_r - y_l) / (x_r - x_l)
            yds[i] = yd
        
        return Linear(xs, yds, False)
