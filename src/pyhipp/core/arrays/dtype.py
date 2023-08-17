from __future__ import annotations
from typing import Union
import numpy as np
from ..abc import HasDictRepr

class DType(HasDictRepr):
    
    _Impl = np.dtype
    
    def __init__(self, dtype: Union[str, DType, _Impl], **kw) -> None:
        
        super().__init__(**kw)
        
        if isinstance(dtype, DType):
            dtype = dtype._impl
        
        self._impl = DType._Impl(dtype)

    @staticmethod
    def int32():
        return DType(np.int32)
    
    @staticmethod
    def int64():
        return DType(np.int64)
    
    @staticmethod
    def float32():
        return DType(np.float32)
    
    @staticmethod
    def float64():
        return DType(np.float64)

    @property
    def alignment(self) -> int:
        return self._impl.alignment
    
    @property
    def byte_order(self) -> str:
        return self._impl.byteorder
    
    @property
    def size(self) -> int:
        return self._impl.itemsize

    def to_simple_repr(self) -> dict:
        return super().to_simple_repr() | {
            'size':         self.size,
            'alignment':    self.alignment,
            'byte_order':   self.byte_order,
            '_impl':        repr(self._impl),
        }