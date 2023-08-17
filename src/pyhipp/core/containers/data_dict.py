from __future__ import annotations
from collections.abc import MutableMapping, Mapping
from typing import Iterator, Union, Any, Tuple
from ..abc import HasSimpleRepr

class DataDict(MutableMapping, HasSimpleRepr):
    '''
    A DataDict is like a Python built-in dict, but allows item access with a 
    tuple of keys and nested keys.
    e.g., 
    d = DataDict({'a': 1, 'b': 2, 'c': DataDict({'d': 3})})
    d['a', 'b', 'c/d']         # => (1, 2, 3)
    
    @init_dict: either None (for an empty dict), or a Mapping object whose items 
    are copied.
    '''
    def __init__(self, dict: Mapping = None):
        super().__init__()
        
        if dict is None:
            dict = {}
        assert isinstance(dict, Mapping)
        self._dict = {k: v for k, v in dict.items()}
    
    def __getitem__(self, key) -> Union[Any, Tuple[Any, ...]]:
        '''
        @key: str | tuple of str.
        
        If `key` is a tuple, return a tuple.
        
        `key` could be a slash-separated name, i.e., a/b/foo, then the name 
        look-up is iteratively performed, i.e., self['a']['b']['foo'] is 
        returned.
        '''
        return DataDict.__call_on_each_tuple_item(
            self.__get_items, self.__get_one_item, key)
        
    def __setitem__(self, key, val) -> None:
        '''Update self with key and val.
        
        If `key` is a tuple, zip(key, val) are iteratively used for update.
        
        `key` could be a slash-separated name, see `__getitem__()`.
        '''
        DataDict.__call_on_each_tuple_item(
            self.__set_items, self.__set_one_item, key, val)

    def __delitem__(self, key) -> None:
        '''
        Delete an item keyed `key`.
        
        If key is a tuple, each item is treated as a single key whose 
        corresponding item is deleted.
        
        `key` could be a slash-separated name, see `__getitem__()`.
        '''
        DataDict.__call_on_each_tuple_item(
            self.__del_items, self.__del_one_item, key)
            
    def __iter__(self) -> Iterator:
        return iter(self._dict)
    
    def __len__(self) -> int:
        return len(self._dict)

    
    # The followings are mixin methods. We overload them for efficiency
    def __contains__(self, key) -> bool:
        '''Only the direct child/element is looked for.'''
        return key in self._dict
    
    def keys(self):
        return self._dict.keys()
    
    def values(self):
        return self._dict.values()
    
    def items(self):
        return self._dict.items()

    def update(self, other: Mapping) -> None:
        
        assert isinstance(other, Mapping)
        
        self._dict.update(other.items())
        
    def clear(self) -> None:
        self._dict.clear()


    # Other useful methods
    def __ior__(self, other) -> DataDict:
        
        '''Update self with another dict | DataDict. 
        Equivalent to self.update(other).'''
        self.update(other)
        
        return self
    
    def to_simple_repr(self) -> Any:
        out = {}
        for k, v in self.items():
            if isinstance(v, HasSimpleRepr):
                out[k] = v.to_simple_repr()
            else:
                out[k] = v
        return out
        
    def get_dict(self) -> dict:
        return self._dict

    def copy(self) -> DataDict:
        return DataDict(self._dict)


    # Implementation Details            
    def __get_one_item(self, key):
        if '/' not in key:
            return self._dict[key]
        
        keys = DataDict.__split_key(key)
        assert len(keys) > 0, f'Empty key {key}'
        
        v = self._dict[keys[0]]
        for k in keys[1:]:
            v = v[k]
        return v
    
    def __get_items(self, keys):
        return tuple( self.__get_one_item(key) for key in keys )
    
    def __set_one_item(self, key, val):
        if '/' not in key:
            self._dict[key] = val
            return
        
        keys = DataDict.__split_key(key)
        assert len(keys) > 0, f'empty key {key}'
        
        v = self._dict
        for k in keys[:-1]:
            v = v[k]
        v[keys[-1]] = val
        
    def __set_items(self, keys, vals):
        assert len(keys) == len(vals), (f'Length of keys and values'
            f' are not equal ({len(keys)} and {len(vals)})')
        
        for k, v in zip(keys, vals):
            self.__set_one_item(k, v)
        
    def __del_one_item(self, key):
        if '/' not in key:
            del self._dict[key]
            return
        
        keys = DataDict.__split_key(key)
        assert len(keys) > 0, f'empty key {key}'
        
        v = self._dict
        for k in keys[:-1]:
            v = v[k]
        del v[ keys[-1] ]
        
    def __del_items(self, keys):
        for k in keys:
            self.__del_one_item(k)
        
    @staticmethod
    def __split_key(key):
        return tuple(k for k in key.split('/') if len(k) > 0)
    
    @staticmethod
    def __call_on_each_tuple_item(fn, fn_fallback, arg0, *args):
        if isinstance(arg0, tuple):
            return fn(arg0, *args)
        else:
            return fn_fallback(arg0, *args)