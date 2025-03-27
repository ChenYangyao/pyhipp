'''
IO library for the Fortran binary format.
'''

from __future__ import annotations
import typing
from typing import Self
from ...core.abc import HasDictRepr
from pathlib import Path
import numpy as np


class BinaryFile(HasDictRepr):

    repr_attr_keys = ('file', 'file_path', 'mode', 'open_kw')

    def __init__(self, file_path: Path, mode='rb', open_kw={}):

        self._file = open(file_path, mode, **open_kw)
        self._file_path = file_path
        self._mode = mode
        self._open_kw = open_kw

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._file.close()

    def seek(self, offset: int, whence=0):
        return self._file.seek(offset, whence)

    def load_bytes(self, n: int = -1):
        return self._file.read(n)

    def skip_rec(self, dtype: np.dtype, n: int = None):
        self.seek(self._rec_size(dtype, n), 1)

    def load_rec(self, dtype: np.dtype, n: int = None):
        if n is None:
            return self.load_rec(dtype, n=1)[0]
        return np.fromfile(self._file, dtype=dtype, count=n)

    def skip_sect(self, dtype: np.dtype, n: int = None):
        n1 = self.load_rec(np.int32)
        self.skip_rec(dtype, n=n)
        n2 = self.load_rec(np.int32)
        assert n1 == n2

    def load_sect(self, dtype: np.dtype, n: int = None) -> np.ndarray:
        n1 = self.load_rec(np.int32)
        data = self.load_rec(dtype, n=n)
        n2 = self.load_rec(np.int32)
        assert n1 == n2
        return data

    def _rec_size(self, dtype: np.dtype, n: int = None):
        '''
        Return the size (no. of bytes) of a record of the given `dtype` and 
        length `n`.
        
        @n: int | None (for an scalar): number of records.
        '''
        if n is None:
            n = 1
        return dtype.itemsize * n
