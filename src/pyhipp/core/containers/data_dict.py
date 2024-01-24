from __future__ import annotations
from collections.abc import MutableMapping, Mapping
from typing import Iterator, Union, Any, Tuple, TypeVar, Hashable, Generic
from ..abc import HasSimpleRepr

Key = TypeVar('Key')
Value = TypeVar('Value')


class DataDict(MutableMapping, HasSimpleRepr, Generic[Key, Value]):
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

    def __getitem__(self,
                    key: Key | tuple[Key, ...]) -> Value | tuple[Value, ...]:
        '''
        @key: str | tuple of str.
        
        If `key` is a tuple, return a tuple.
        
        `key` could be a slash-separated name, i.e., a/b/foo, then the name 
        look-up is iteratively performed, i.e., self['a']['b']['foo'] is 
        returned.
        '''

        if isinstance(key, tuple):
            return tuple(self[_key] for _key in key)

        if '/' not in key:
            return self._dict[key]

        keys = DataDict.__split_key(key)
        assert len(keys) > 0, f'Empty key {key}'

        v = self._dict[keys[0]]
        for k in keys[1:]:
            v = v[k]
        return v

    def __setitem__(
            self, key: Key | tuple[Key, ...],
            val: Value | tuple[Value, ...]) -> None:
        '''Update self with key and val.
        
        If `key` is a tuple, zip(key, val) are iteratively used for update.
        
        `key` could be a slash-separated name, see `__getitem__()`.
        '''

        # set items by keys and values
        if isinstance(key, tuple):
            assert len(key) == len(val), \
                (f'Lengths of keys and values'
                 f' are not equal ({len(key)} and {len(val)})')
            for _key, _val in zip(key, val):
                self[_key] = _val
            return

        # or, by a single pair of key and value.
        if '/' not in key:
            self._dict[key] = val
            return

        keys = DataDict.__split_key(key)
        assert len(keys) > 0, f'empty key {key}'

        v = self._dict
        for k in keys[:-1]:
            v = v[k]
        v[keys[-1]] = val

    def __delitem__(self, key: Key | tuple[Key, ...]) -> None:
        '''
        Delete an item keyed `key`.
        
        If key is a tuple, each item is treated as a single key whose 
        corresponding item is deleted.
        
        `key` could be a slash-separated name, see `__getitem__()`.
        '''
        if isinstance(key, tuple):
            for _key in key:
                del self[_key]
            return

        if '/' not in key:
            del self._dict[key]
            return

        keys = DataDict.__split_key(key)
        assert len(keys) > 0, f'empty key {key}'

        v = self._dict
        for k in keys[:-1]:
            v = v[k]
        del v[keys[-1]]

    def __iter__(self) -> Iterator:
        return iter(self._dict)

    def __len__(self) -> int:
        return len(self._dict)

    # The followings are mixin methods. We overload them for efficiency

    def __contains__(self, key) -> bool:
        '''Only the direct child/element is looked for.'''
        return key in self._dict

    def keys(self) -> Iterator[Key]:
        return self._dict.keys()

    def values(self) -> Iterator[Value]:
        return self._dict.values()

    def items(self) -> Iterator[Tuple[Key, Value]]:
        return self._dict.items()

    def update(self, other: Mapping) -> None:

        assert isinstance(other, Mapping)

        self._dict.update(other.items())

    def clear(self) -> None:
        self._dict.clear()

    # Other useful methods

    def __ior__(self, other: Mapping) -> DataDict:
        '''Update self with another dict | DataDict. 
        Equivalent to self.update(other).'''
        self.update(other)

        return self

    def __or__(self, other: Mapping) -> DataDict:
        '''Return a new DataDict that is the union of self and other.'''
        out = self.copy()
        out |= other
        return out

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

    @staticmethod
    def __split_key(key):
        return tuple(k for k in key.split('/') if len(k) > 0)
