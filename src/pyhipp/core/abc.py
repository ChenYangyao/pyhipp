from __future__ import annotations
from typing import Any, Callable, Dict, Tuple, List, Any
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


class _ChainLogger:
    def __init__(self, obj: HasLog, 
                 wrap_on = 5, flush=True, indent=4):
        self.obj = obj
        self.wrap_on = wrap_on
        self.flush = flush
        self.indent = ' '*indent
        self.count = 0
    
    def __call__(self, *args):
        self.count += 1
        start = self.indent if self.count % self.wrap_on == 1 else ''
        end = ',\n' if self.count % self.wrap_on == 0 else ', '
        self.obj.log(start, *args, flush=self.flush, end=end, named=False)
        
    def __enter__(self):
        return self
    
    def __exit__(self, *exc):
        text = ''
        if self.count % self.wrap_on != 0:
            text = '\n'
        text += self.indent + f'({self.count} items done)'
        self.obj.log(text, flush=self.flush, named=False)

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
        return self.log_for(
            self, *args, end=end, sep=sep, flush=flush, named=named,
            timed=timed, time_fmt=time_fmt)

    def __get_obj_name(self, obj: str | HasName | Any):
        if isinstance(obj, str):
            type_name = obj
        elif isinstance(obj, HasName):
            type_name = obj.name
        else:
            type_name = type(obj).__name__

    def log_for(self, obj, *args, end='\n', sep='', flush=True, named=True,
                timed=False, time_fmt='%Y-%m-%d %H:%M:%S') -> None:
        kw = dict(end=end, sep=sep, flush=flush)
        if self.verbose:
            prefix = ''
            if named:
                type_name = self.__get_obj_name(obj)        
                prefix += f'[{type_name}]'
            if timed:
                t = strftime(time_fmt, localtime())
                prefix += f'[{t}]'
            if len(prefix) > 0:
                print(prefix, ' ', *args, **kw)
            else:
                print(*args, **kw)
        return self
    
    def for_obj(self, obj):
        return HasLog(verbose=self.verbose, name=self.__get_obj_name(obj))
    
    def chain(self, wrap_on=5):
        '''
        Examples
        --------
        log = abc.HasLog(True, name='Worker')

        log.log('starting the chain of works')
        with log.chain() as c:
            for i in range(4):
                c(i)
            for i in range(3):
                c(i)
                
        log.log('starting the second chain of works')
        with log.chain() as c:
            c('Initialized')
            c('Updated')
            c('Finished')
        
        Output:
        [Worker] starting the chain of works
            0, 1, 2, 3, 0,
            1, 2,
            (7 items done)
        [Worker] starting the second chain of works
            Initialized, Updated, Finished,
            (3 items done)
        '''
        return _ChainLogger(self, wrap_on=wrap_on)


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
