from __future__ import annotations
from typing import Any, Callable, Dict, Tuple, List
import numpy as np
import yaml
import json
from time import localtime, strftime
import pprint


class IsImmutable:
    def __init__(self, **kw) -> None:
        super().__init__(**kw)


class HasSimpleRepr:
    def __init__(self, **kw) -> None:
        super().__init__(**kw)

    def __repr__(self) -> str:
        return pprint.pformat(
            self.to_simple_repr(),
            sort_dicts=False, compact=True, indent=2)

    def to_simple_repr(self) -> Any:
        return object.__repr__(self)

    def to_yaml(self, **kw) -> str:
        '''
        kw: e.g., indent.
        '''
        return yaml.dump(self.to_simple_repr(), **kw)

    def to_json(self, **kw) -> str:
        '''
        kw: e.g., indent.
        '''
        return json.dumps(self.to_simple_repr(), **kw)


class HasDictRepr(HasSimpleRepr):

    repr_enable_type_string: bool = True
    repr_attr_keys: Tuple[str] = ()

    def to_simple_repr(self) -> dict:
        out = {}
        if self.repr_enable_type_string:
            out['type'] = self.__class__.__name__
        for k in self.repr_attr_keys:
            v = getattr(self, k)
            if isinstance(v, HasSimpleRepr):
                v = v.to_simple_repr()
            out[k] = v
        return out


class HasListRepr(HasSimpleRepr):

    repr_enable_type_string: bool = False
    repr_attr_keys: Tuple[str] = ()

    def to_simple_repr(self) -> List:
        attrs = []
        if self.repr_enable_type_string:
            attrs.append(self.__class__.__name__)
        for k in self.repr_attr_keys:
            v = getattr(self, k)
            if isinstance(v, HasSimpleRepr):
                v = v.to_simple_repr()
            attrs.append(v)
        return attrs


class HasListRepr(HasListRepr):

    def to_simple_repr(self) -> Tuple:
        return tuple(super().to_simple_repr())


class HasName:
    def __init__(self, name: str = None, **kw) -> None:
        super().__init__(**kw)

        if name is None:
            name = type(self).__name__

        self.name = str(name)


class HasLog(HasName):
    def __init__(self, verbose=False, **kw) -> None:
        super().__init__(**kw)

        self.verbose = verbose

    def verbose_on(self) -> None:
        self.verbose = True

    def verbose_off(self) -> None:
        self.verbose = False

    def log(self, *args, end='\n', sep='', flush=True, named=True,
            timed=False, time_fmt='%Y-%m-%d %H:%M:%S') -> None:
        kw = dict(end=end, sep=sep, flush=flush)
        if self.verbose:
            prefix = ''
            if named:
                prefix += f'[{self.name}]'
            if timed:
                t = strftime(time_fmt, localtime())
                prefix += f'[{t}]'
            if len(prefix) > 0:
                print(prefix, ' ', *args, **kw)
            else:
                print(*args, **kw)
        return self


class HasValue:
    def __init__(self, value: np.ndarray, copy=True, **kw) -> None:
        super().__init__(**kw)

        self.value = np.array(value, copy=copy)

    def set_value(self, value: np.ndarray) -> None:
        self.value[...] = value


class HasCache:
    def __init__(self) -> None:
        self.cache: Dict[Any, Any] = {}

    def get_cache_or(self, key: Any, fallback: Callable) -> Any:
        c = self.cache
        if key in c:
            out = c[key]
        else:
            out = fallback()
            c[key] = out

        return out

    def put_cache(self, key: Any, value: Any) -> None:
        self.cache[key] = value


class HasMultiCache:
    def __init__(self) -> None:
        self.cache: Dict[str, Dict[str, Any]] = {}

    def get_cache_or(self, key: str, subkey: str, fallback: Callable) -> Any:
        if key not in self.cache:
            self.cache[key] = {}

        c = self.cache[key]
        if subkey in c:
            out = c[subkey]
        else:
            out = fallback()
            c[subkey] = out

        return out
