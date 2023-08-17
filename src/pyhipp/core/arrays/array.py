from __future__ import annotations
import numpy as np
from typing import Sequence, Union

from .dtype import DType
from ..abc import HasDictRepr
from .dtype import DType


class Array(HasDictRepr):
    
    _Impl = np.ndarray
    ArrayLike = Union[_Impl, Sequence, float, int]
    
    def __init__(self, data: Union[Array, ArrayLike], dtype: DType = None, 
                 copy: bool = True, **kw) -> None:
        
        super().__init__(**kw)
        
        if isinstance(dtype, DType):
            dtype = dtype._impl
        if isinstance(data, Array):
            data = data._impl
            
        self._impl = np.array(data, dtype=dtype, copy=copy)
        
    @classmethod
    def zeros(cls, shape: Shape, dtype: DType = None, **kw) -> Array:
        shape, dtype = Array.__parse_descr(shape, dtype)
        return cls(np.zeros(shape, dtype=dtype), copy=False, **kw)
    
    @classmethod
    def ones(cls, shape: Shape, dtype: DType = None, **kw) -> Array:
        shape, dtype = Array.__parse_descr(shape, dtype)
        return cls(np.ones(shape, dtype=dtype), copy=False, **kw)
        
    def copied(self) -> Array:    
        return Array(self._impl.copy(), copy=False)
    
    def flatten(self, copy=True) -> Array:
        out = self._impl.flatten() if copy else self._impl.ravel()
        return Array(out, copy=False)
    
    def sorted(self, axis = None, kind = None) -> Array:
        out = np.sort(self._impl, axis=axis, kind=kind)
        return Array(out, copy=False)
    
    def argsort(self, axis = None, kind = None) \
            -> Union[FlatIndexArray, IntArray]:
        out = self._impl.argsort(axis=axis, kind=kind)
        if axis is None:
            out = FlatIndexArray(out, self.size, copy=False)
        else:
            out = IntArray(out, copy=False)
        return out
    
    @property
    def n_dims(self) -> int:
        return self._impl.ndim
    
    @property
    def shape(self) -> Shape:
        return Shape(self._impl.shape)
    
    @property
    def size(self) -> int:
        return self._impl.size
    
    @property
    def dtype(self) -> DType:
        return DType(self._impl.dtype)
        
    def __len__(self) -> int:
        return len(self._impl)
    
    def to_simple_repr(self) -> dict:
        data = self._impl
        return super().to_simple_repr() | {
            'n_dims':   self.n_dims,
            'shape':    tuple(self.shape._impl), 
            'dtype':    str(self.dtype._impl),
            'data':     data,
        }
        
    @staticmethod
    def __parse_descr(shape: Shape, dtype: DType):
        if np.isscalar(shape):
            shape = (shape,)
        shape = Shape(shape)._impl
        if dtype is not None:
            dtype = DType(dtype)._impl
        return shape, dtype
        
class IntArray(Array):
    pass

class BoolArray(Array):
    pass

class Shape(IntArray):

    def __init__(self, *args, **kw) -> None:
        
        super().__init__(*args, **kw)    
        
        assert self.n_dims == 1
        
    

class FlatIndexArray(IntArray):
    
    def __init__(self, n_indexed: int, *args, **kw) -> None:
        
        super().__init__(*args, **kw)
        
        self.n_indexed = int(n_indexed)
        
        assert self.n_dims == 1
        
class IndexArray(IntArray):
    
    def __init__(self, shape_indexed: Shape, *args, **kw) -> None:
        
        super().__init__(*args, **kw)
        
        self.shape_indexed = Shape(shape_indexed)
        
        assert self.n_dims == 2
        assert self.shape[-1] == len(self.shape_indexed)

