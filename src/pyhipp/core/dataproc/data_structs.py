from collections.abc import Mapping
import numpy as np

class Dict:
    def __init__(self, *dicts: Mapping, **items):
        self.d = Dict.combine_dicts(*dicts, **items)
        
    @staticmethod
    def combine_dicts(*dicts: Mapping, **items):
        out = {}
        for d in dicts:
            if d is not None:
                out.update(d)
        out.update(items)
        return out
    
    @staticmethod
    def none_to_empty(d: Mapping = None):
        if d is None:
            d = {}
        return d
    
class Array:
    @staticmethod
    def as_array(*args, copy=True):
        '''
        For each x in args, convert it to np.ndarray.
        If x is None, get None.
        
        @copy: True then produce a copy.
        
        If len(args) == 1, return a single np.ndarray, otherwise a tuple.
        '''
        n_args = len(args)
        assert n_args >= 1
        
        if n_args == 1:
            out = Array.__as_array_one(args[0], copy)
        else:
            out = tuple(Array.__as_array_one(arg, copy) for arg in args) 
        return out
    
    @staticmethod
    def __as_array_one(x, copy):
        if x is None:
            return None
        return np.array(x, copy=copy)

    
    @staticmethod
    def count_to_offset(count, has_end=True):
        '''
        Return the `offset` array from `count` array.
        
        @has_end: bool. If true, `offset` includes the ending index, 
        i.e., sum(count).
        '''
        
        off = np.concatenate(( (0,), np.cumsum(count) ))
        if not has_end:
            off = off[:-1]
        return off
    
    @staticmethod
    def count_to_offset_pairs(count) -> np.ndarray:
        off = Array.count_to_offset(count)
        return np.stack((off[:-1], off[1:]), axis=-1)