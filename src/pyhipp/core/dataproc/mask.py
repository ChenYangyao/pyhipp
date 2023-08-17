import numpy as np
from typing import Any, MutableMapping

class Mask:
    @staticmethod
    def put(x_dst: np.ndarray, x_src: np.ndarray, mask: np.ndarray) -> None:
        '''
        ``mask`` is repeated if less than length of ``x_dst``.
        '''
        np.putmask(x_dst, mask, x_src)
        
    @staticmethod
    def zip(masks, arrays, default=None):
        '''
        Masked zip.
        out[i] = arrays[m][i] if masks[m][i] == True.
        For multiple True at i, taken from the first. For none True at i, 
        take ``default``.
        '''
        kw = {}
        if default is not None:
            kw['default'] = default
        return np.select(masks, arrays, **kw)
    
    @staticmethod
    def bool_to_idx(bool_array: np.ndarray):
        bool_array = np.asarray(bool_array)
        assert bool_array.ndim == 1
        return bool_array.nonzero()[0]
    
    @staticmethod
    def idx_to_bool(n: int, idx_array: np.ndarray):
        idx_array = np.asarray(idx_array)
        assert idx_array.ndim == 1
        
        bool_array = np.zeros(n, dtype=bool)
        bool_array[idx_array] = True
        
        return bool_array
    
    @staticmethod
    def on_dict(d: MutableMapping, mask: Any, copy: bool = False):
        out_d = type(d)()
        for k, v in d.items():
            out_d[k] = Mask.on_array(mask, v, copy=copy)
        return out_d
    
    @staticmethod
    def on_array(mask: Any, a: np.ndarray, copy: bool = False):
        return np.array(a[mask], copy=copy)
        
    @staticmethod
    def on_arrays(mask: Any, 
                  a: np.ndarray, 
                  *other_a: np.ndarray, 
                  copy: bool = False):
        out = Mask.on_array(mask, a, copy=copy)
        
        if len(other_a) == 0:
            return out
        
        return (out,) + tuple(Mask.on_array(mask, _a, copy=copy) 
                              for _a in other_a)
        