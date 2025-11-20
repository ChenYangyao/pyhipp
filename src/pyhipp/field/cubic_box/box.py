from __future__ import annotations
import typing
import numpy as np
from pyhipp.core import abc
from numba.experimental import jitclass
import numba


@jitclass
class _PeriodicBox:

    l_box: numba.float64

    def __init__(self, l_box,):
        self.l_box = l_box

    def ishift_in(self, x: np.ndarray) -> None:
        l_box = self.l_box
        x[x < 0.] += l_box
        x[x >= l_box] -= l_box
        assert np.all(x >= 0.)
        assert np.all(x < self.l_box)

    def ishift_to(self, x: np.ndarray, x_ref: np.ndarray) -> None:
        l_box = self.l_box
        l_half = .5 * l_box
        dx = x - x_ref
        x[dx < -l_half] += l_box
        x[dx >= l_half] -= l_box

        dx = x - x_ref
        assert np.all(dx >= -l_half)
        assert np.all(dx < l_half)

    def shift_to(self, x: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
        x = x.copy()
        self.ishift_to(x, x_ref)
        return x


class PeriodicBox(abc.HasDictRepr):

    repr_attr_keys = ('l_box', )

    def __init__(self, l_box, **kw):
        super().__init__(**kw)
        self.l_box = l_box

    def ishift_in(self, x: np.ndarray) -> None:
        l_box = self.l_box
        x[x < 0.] += l_box
        x[x >= l_box] -= l_box
        assert np.all(x >= 0.)
        assert np.all(x < self.l_box)

    def ishift_to(self, x: np.ndarray, x_ref: np.ndarray) -> None:
        l_box = self.l_box
        l_half = .5 * l_box
        dx = x - x_ref
        x[dx < -l_half] += l_box
        x[dx >= l_half] -= l_box

        dx = x - x_ref
        assert np.all(dx >= -l_half)
        assert np.all(dx < l_half)
