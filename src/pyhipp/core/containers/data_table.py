from __future__ import annotations
from collections.abc import MutableMapping, Mapping
from collections import OrderedDict
from typing import Iterator, Union, Any, Tuple, TypeVar, Hashable, Generic, Self
from ..abc import HasSimpleRepr
import numpy as np
import pandas as pd

Key = str
Keys = tuple[Key, ...]
KeyOrKeys = Key | Keys

Value = np.ndarray
Values = tuple[Value, ...]
ValueOrValues = Value | Values


class _ILoc(HasSimpleRepr):

    def __init__(self, ref: DataTable) -> None:
        self._ref = ref

    def __getitem__(self, args: np.ndarray | slice) -> DataTable:
        return self._ref.subset(args)


class DataTable(HasSimpleRepr, MutableMapping[Key, Value]):
    '''
    An ordered dictionary specifically designed for tabular data. Each item 
    represents a column of the table, with a key typed as str and a value
    typed as np.ndarray.
    
    Initializer
    -----------
    @data: a Mapping of key-value pairs.
    @copy: whether to copy the values.
    
    Notes
    -----
    It is recommended, but not mandatory, that all values have the same length.
    Methods such as stackings and concatenations assume this constraint.
    
    Each value can be M-D array, with M >= 1.
    '''

    def __init__(self, data: Mapping[Key, Value] = None, copy=True):
        super().__init__()

        if data is None:
            data = {}

        assert isinstance(data, Mapping)

        self._data = OrderedDict({
            k: np.array(v, copy=copy) for k, v in data.items()
        })

    def __getitem__(self, key: KeyOrKeys) -> ValueOrValues:
        '''
        @key: str | tuple of str. If `key` is a tuple, return a tuple of 
        values, each corresponding to an item in `key`.
        Each returned value is a reference to the column data.
        '''
        if isinstance(key, tuple):
            return tuple(self[_key] for _key in key)
        return self._data[key]

    def __setitem__(self, key: KeyOrKeys, val: ValueOrValues) -> None:
        '''
        Update self with key and val.
        If `key` is a tuple, zip(key, val) are iteratively used for update.
        Values are deep copied.
        '''
        _data = self._data

        if not isinstance(key, tuple):
            _data[key] = np.array(val)
            return

        assert len(key) == len(val), (
            f'Lengths of keys and values'
            f' are not equal ({len(key)} and {len(val)})')
        for _key, _val in zip(key, val):
            _data[_key] = np.array(_val)

    def __delitem__(self, key: KeyOrKeys) -> None:
        '''
        Delete an item keyed `key`.
        If key is a tuple, each item is treated as a single key whose 
        corresponding item is deleted.
        
        Examples 
        --------
        ```py
        del dt['c']
        del dt['a', 'b']
        ```
        
        '''
        _data = self._data

        if not isinstance(key, tuple):
            del _data[key]
            return

        for _key in key:
            del _data[_key]

    def __iter__(self) -> Iterator[Key, Value]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    # The followings are mixin methods. We overload them for efficiency
    def __contains__(self, key: Key) -> bool:
        return key in self._data

    def keys(self) -> Iterator[Key]:
        return self._data.keys()

    def values(self) -> Iterator[Value]:
        return self._data.values()

    def items(self) -> Iterator[Tuple[Key, Value]]:
        return self._data.items()

    def update(self, other: Mapping[Key, Value], copy=True) -> None:
        self._data.update({
            k: np.array(v, copy=copy) for k, v in other.items()
        })

    def clear(self) -> None:
        self._data.clear()

    # For attribute access
    def __getattr__(self, key: Key) -> Value:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f'Attribute {key} not found')

    def __dir__(self) -> list[str]:
        return list(self.keys()) + list(super().__dir__())

    # Other useful methods
    def __ior__(self, other: Mapping[Key, Value]) -> Self:
        '''
        Update self with another Mapping. 
        Equivalent to self.update(other).
        '''
        self.update(other)
        return self

    def __or__(self, other: Mapping[Key, Value]) -> DataTable:
        '''Return a new DataTable that is the union of self and other.'''
        out = self.copy()
        out |= other
        return out

    def copy(self) -> DataTable:
        return DataTable(self._data)

    def to_simple_repr(self) -> Any:
        return self.to_dict(copy=False)

    def _to_h5_data(self) -> dict:
        return self.to_dict(copy=False)

    @classmethod
    def _from_h5_data(cls, data: dict, **init_kw) -> Self:
        init_kw = {
            'data': data,
            'copy': False
        } | init_kw
        return cls(**init_kw)

    def to_dict(self, copy=True) -> dict[str, np.ndarray]:
        return {k: np.array(v, copy=copy) for k, v in self.items()}

    def to_dict_of_list(self) -> dict[str, list]:
        return {k: v.tolist() for k, v in self.items()}

    def col_stacked(self) -> np.ndarray:
        return np.column_stack(list(self.values()))

    def row_stacked(self) -> np.ndarray:
        return np.row_stack(list(self.values()))

    def concat(self, *other: Mapping[Key, Value]) -> DataTable:
        '''
        Concatenate self with other DataTable by concatenating (along axis 0) 
        their values with the same key. Return a new DataTable.
        
        Examples
        --------
        ```py
        dt = DataTable({
            'a': np.arange(4),
            'b': np.linspace(0., 1., 4),
            'c': np.random.rand(4, 3),
        })
        dt
        ```
        Output:
        { 'a': array([0, 1, 2, 3]),
          'b': array([0.        , 0.33333333, 0.66666667, 1.        ]),
          'c': array([[0.33471348, 0.17180129, 0.48101257],
            [0.04147932, 0.26516706, 0.98233573],
            [0.75800727, 0.67616934, 0.37048391],
            [0.03942748, 0.88630544, 0.78747978]])}
        
        ```py
        dt.concat(dt)
        ```
        Output:
        { 'a': array([0, 1, 2, 3, 0, 1, 2, 3]),
          'b': array([0.        , 0.33333333, 0.66666667, 1.        , 0.        ,
               0.33333333, 0.66666667, 1.        ]),
          'c': array([[0.69407056, 0.36290328, 0.48621282],
               [0.50178046, 0.18085208, 0.50465674],
               [0.99053128, 0.20363366, 0.96262781],
               [0.29595584, 0.35776469, 0.70314595],
               [0.69407056, 0.36290328, 0.48621282],
               [0.50178046, 0.18085208, 0.50465674],
               [0.99053128, 0.20363366, 0.96262781],
               [0.29595584, 0.35776469, 0.70314595]])}
        '''
        all_tables = [self] + list(other)
        return DataTable({
            k: np.concatenate([t[k] for t in all_tables])
            for k in self.keys()
        })

    def subset(self, args: np.ndarray | slice, copy=True) -> DataTable:
        '''
        Return a new DataTable that is a subset (rows) of self, with each column 
        selected by `args`.
        '''
        return DataTable({
            k: v[args] for k, v in self.items()
        }, copy=copy)

    def subcol(self, keys: Keys, copy=True) -> DataTable:
        '''
        Return a new DataTable that is a subset (columns) of self.
        '''
        return DataTable({
            k: self[k] for k in keys
        }, copy=copy)

    @property
    def iloc(self) -> _ILoc:
        return _ILoc(self)
