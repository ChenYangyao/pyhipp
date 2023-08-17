from __future__ import annotations
import h5py
from typing import Union, Iterable, Iterator, List, Tuple, Any, Mapping
from ...core.abc import HasSimpleRepr
import re
from collections.abc import Sequence
import numpy as np
import re

class Utils:
    Key = str
    KeyOrKeys = Union[str, Iterable[str]]
    ArrayOrAny = Union[np.ndarray, Any]
    ArrayOrArrays = Union[np.ndarray, Tuple[np.ndarray, ...], Any]
    SupportedData = Union[bytes, str, np.ndarray, Any]
    SupportedDataFromLoad = Union[bytes, np.ndarray, Any]
    
    @staticmethod
    def find_matched_keys(keys: Iterable[str], 
                          re_pattern: str = None) -> List[str]:
        if re_pattern is None:
            return list(keys)
        re_obj = re.compile(re_pattern)
        return [key for key in keys if re_obj.match(key)]
    
    @staticmethod
    def is_supported_key(key: str):
        return isinstance(key, str)


    @staticmethod
    def is_supported_data(data: Any) -> bool:
        return isinstance(data, (str, bytes, np.ndarray)) \
            or np.isscalar(data)
    
    @staticmethod
    def is_data_dict_supported(data_dict: Mapping) -> bool:
        Self = Utils
        if not isinstance(data_dict, Mapping):
            return False, 'not a Mapping'
        for k, v in data_dict.items():
            if not Self.is_supported_key(k):
                return False, f'key {k}'    
            if isinstance(v, Mapping):
                res, hint = Self.is_data_dict_supported(v)
                if not res:
                    return False, f'{hint} for key {k}'
            elif not Self.is_supported_data(v):
                return False, f'value {v} for key {k}'
        return True, None

class Obj:
    '''
    Based type for all HDF5 object types in this module.
    '''
    def __init__(self, raw: Any = None, **kw) -> None:
        super().__init__(**kw)
        self._raw = raw
        
    def ls(self, file = None, flush = True, **what_kw) -> None:
        print(self.what(**what_kw), file=file, flush=flush)
        
    def what(self, **kw) -> str:
        raise NotImplementedError()
        
class KeyList(Sequence, HasSimpleRepr):
    def __init__(self, keys: Iterable[str]):
        self._keys = tuple(keys)
        
    def __getitem__(self, i: int) -> str:
        return self._keys[i]
    
    def __len__(self) -> int:
        return len(self._keys)
    
    def __contains__(self, key: str) -> bool:
        return key in self._keys
    
    def __reversed__(self) -> Iterator[str]:
        return reversed(self._keys)
    
    def __iter__(self) -> Iterator[str]:
        return iter(self._keys)
        
    def matched(self, key_re: str = None) -> KeyList:
        return KeyList(Utils.find_matched_keys(self._keys, key_re))
        
    def to_simple_repr(self) -> Tuple[str, ...]:
        return self._keys
    
