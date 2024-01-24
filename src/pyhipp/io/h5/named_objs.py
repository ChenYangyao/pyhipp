from __future__ import annotations
from collections.abc import Iterable
import h5py
from typing import Union, Iterable, Iterator, List, Tuple, Any, Mapping, Literal
from ...core import DataDict
from pathlib import Path
import numpy as np
from .utils import Obj, KeyList, Utils


class AttrManager(Obj):

    Raw = h5py.AttributeManager

    def __init__(self, raw: Raw = None, **kw) -> None:
        super().__init__(raw=raw, **kw)
        self._raw: AttrManager.Raw

    def __getitem__(self, key: Utils.KeyOrKeys) -> Utils.ArrayOrArrays:
        if isinstance(key, str):
            return self._raw[key]
        else:
            return tuple(self[k] for k in key)

    def __setitem__(self, key: str, value: Utils.SupportedData) -> None:
        '''
        Change the value of an existing attribute while preserving its datatype.
        Raise a KeyError if the attribute does not exist.
        '''
        if key not in self:
            raise KeyError(f'Attribute {key} not found')
        self._raw.modify(key, value)

    def __len__(self) -> int:
        return len(self._raw)

    def __contains__(self, key) -> bool:
        return key in self._raw

    def __getattr__(self, key: str) -> Utils.ArrayOrAny:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f'Attribute {key} not found')

    def __dir__(self) -> Iterable[str]:
        return list(super().__dir__()) + list(self.__keys())

    def create(self, key, data, flag='x') -> None:
        '''
        Create a attribute with given data.
        @flag: specify how to deal with exacting attribute keyed ``key``, 
               one of:
               'x': raise ValueError.
               'ac' or 'ca': overwrite it.
        '''
        if flag not in ('x', 'ac', 'ca'):
            raise ValueError(f'Invalid argument {flag=}')

        if key in self:
            if flag == 'x':
                raise ValueError(f'Redundant key {key}')
            self._raw.modify(key, data)
        else:
            self._raw.create(key, data=data)

    def create_empty(self, key: str, shape: Tuple[int, ...], dtype: np.dtype,
                     flag='x') -> None:
        '''
        Create an empty attribute with given shape and dtype.
        Use __setitem__() to fill its value later.
        
        @flag: specify how to deal with exacting attribute keyed ``key``, 
               one of:
               'x': raise ValueError.
               'ac' or 'ca': silently return.
        '''
        if flag not in ('x', 'ac', 'ca'):
            raise ValueError(f'Invalid argument {flag=}')

        if key in self:
            if flag == 'x':
                raise ValueError(f'Redundant key {key}')
        else:
            self._raw.create(key, shape=shape, dtype=dtype)

    def keys(self) -> KeyList:
        return KeyList(self.__keys())

    def values(self) -> Iterator[Utils.ArrayOrAny]:
        raw = self._raw
        return (raw[k] for k in self.__keys())

    def items(self) -> Iterator[Tuple[str, Utils.ArrayOrAny]]:
        raw = self._raw
        return ((k, raw[k]) for k in self.__keys())

    def load(self, key_re: str = None, keys: Iterable[str] = None) -> DataDict:
        keys = self.keys() if keys is None else KeyList(keys)
        return DataDict({k: self[k] for k in keys.matched(key_re)})

    def dump(self, data_dict: Mapping, flag='x'):
        for k, v in data_dict.items():
            self.create(k, v, flag=flag)

    def __repr__(self) -> str:
        return f'AttrManager(keys={self.keys()})'

    def what(self) -> str:
        if len(self) == 0:
            return ''
        attrs = tuple(f'{k}={v}' for k, v in self.items())
        return '[' + ', '.join(attrs) + ']'

    def __keys(self) -> Iterable[str]:
        return self._raw.keys()


class DatasetManager(Obj):

    Raw = h5py.Group

    CreateFlag = Literal['x', 'ac', 'ca']

    def __init__(self, raw: Any = None, **kw) -> None:
        super().__init__(raw, **kw)
        self._raw: DatasetManager.Raw

    def __getitem__(self, key: Utils.KeyOrKeys) -> Utils.ArrayOrArrays:
        if isinstance(key, str):
            v = self._raw[key]
            if not isinstance(v, h5py.Dataset):
                raise KeyError(f'Object {key} is not a dataset')
            return v[()]
        else:
            return tuple(self[k] for k in key)

    def __setitem__(self, key: str, value: Utils.SupportedData) -> None:
        '''
        Change the value of an existing attribute while preserving its datatype.
        Raise a KeyError if the attribute does not exist.
        '''
        if key not in self:
            raise KeyError(f'Dataset {key} not found')
        self._raw[key][...] = value

    def __contains__(self, key):
        if key not in self._raw:
            return False
        return isinstance(self._raw[key], h5py.Dataset)

    def __getattr__(self, key: str) -> Utils.ArrayOrAny:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f'Dataset {key} not found')

    def __dir__(self) -> Iterable[str]:
        return list(super().__dir__()) + list(self.__keys())

    def create(self, key, data, flag: CreateFlag = 'x') -> Dataset:
        '''
        Create a dataset with given data.
        
        @flag: if existing dataset keyed ``key``, 
            'x': raise ValueError.
            'ac' or 'ca': overwrite if it is an dataset, otherwise raise 
            ValueError.
        '''
        if flag not in ('x', 'ac', 'ca'):
            raise ValueError(f'Invalid argument {flag=}')

        if key in self._raw.keys():
            if flag == 'x':
                raise ValueError(f'Redundant key {key}')
            val: h5py.Dataset = self._raw[key]
            if not isinstance(val, h5py.Dataset):
                raise ValueError(f'key {key} does not refer to a dataset')
            val[...] = data
        else:
            val = self._raw.create_dataset(key, data=data)

        return Dataset(val)

    def create_empty(self, key: str, shape: Tuple[int, ...], dtype: np.dtype,
                     flag: CreateFlag = 'x') -> None:
        '''
        Create an empty dataset with given shape and dtype. Return the newly 
        created one.
        Use DatasetManager.__setitem__() or Dataset.__setitem__() to fill 
        its value later.
        @flag: specify how to deal with exacting attribute keyed ``key``, 
            one of:
            'x': raise ValueError.
            'ac' or 'ca': open and return the dataset.
        '''
        if flag not in ('x', 'ac', 'ca'):
            raise ValueError(f'Invalid argument {flag=}')

        if key in self:
            if flag == 'x':
                raise ValueError(f'Redundant key {key}')
            val = self._raw[key]
            if not isinstance(val, h5py.Dataset):
                raise ValueError(f'key {key} does not refer to a dataset')
        else:
            val = self._raw.create_dataset(key, shape=shape, dtype=dtype)

        return Dataset(val)

    def keys(self) -> KeyList:
        return KeyList(self.__keys())

    def values(self) -> Iterator[Utils.ArrayOrAny]:
        for _, v in self.__items():
            yield v[()]

    def items(self) -> Iterator[Tuple[str, Utils.ArrayOrAny]]:
        for k, v in self.__items():
            yield k, v[()]

    def load(self, key_re: str = None, keys: Iterable[str] = None) -> DataDict:
        keys = self.keys() if keys is None else KeyList(keys)
        return DataDict({k: self[k] for k in keys.matched(key_re)})

    def dump(self, data_dict: Mapping, flag: CreateFlag = 'x'):
        for k, v in data_dict.items():
            self.create(k, v, flag=flag)

    def __repr__(self) -> str:
        return f'DatasetManager(keys={self.keys()})'

    def what(self) -> str:
        if len(self) == 0:
            return ''
        return '[' + ', '.join(self.__keys()) + ']'

    def __keys(self) -> Iterable[str]:
        return (k for k, _ in self.__items())

    def __items(self) -> Iterable[str, h5py.Dataset]:
        g = self._raw
        Dset = h5py.Dataset
        for k in g.keys():
            v = g[k]
            if isinstance(v, Dset):
                yield k, v


class NamedObj(Obj):
    '''
    Based type for all HDF5 named object types in this module.
    '''

    Raw = Union[h5py.Dataset, h5py.Group]
    Concrete = Union['Dataset', 'Group']

    def __init__(self, raw: Raw = None, **kw) -> None:
        super().__init__(raw=raw, **kw)
        self._raw: NamedObj.Raw

    @property
    def attrs(self) -> AttrManager:
        return AttrManager(self._raw.attrs)

    def __repr__(self) -> str:
        name = type(self).__name__
        return f'{name}(raw={self._raw})'


class Dataset(NamedObj):

    Raw = h5py.Dataset

    def __init__(self, raw: Raw = None, **kw) -> None:
        super().__init__(raw, **kw)
        self._raw: Dataset.Raw

    def __getitem__(self, key) -> Union[np.ndarray, Any]:
        return self._raw[key]

    def __setitem__(self, key, val) -> None:
        self._raw[key] = val

    @property
    def dtype(self):
        return self._raw.dtype

    @property
    def ndim(self):
        return self._raw.ndim

    @property
    def shape(self):
        return self._raw.shape

    def what(self, attr=True) -> str:
        s_shape = f', {self.shape}' if self.ndim > 0 else ''
        out = f'({self.dtype}{s_shape})'
        if not attr:
            return out

        return out + self.attrs.what()


class Group(NamedObj):

    Raw = h5py.Group

    CreateFlag = Literal['x', 'ac', 'ca']
    DumpFlag = CreateFlag | DatasetManager.CreateFlag

    def __init__(self, raw: Raw = None, **kw) -> None:
        super().__init__(raw=raw, **kw)
        self._raw: Group.Raw

    def create_group(self, key, flag: CreateFlag = 'x') -> Group:
        '''
        Create and return a data group.
        
        @key: name of the group.
        @flag: specify how to deal with existing group. One of:
            "x"          - exclusive, i.e., throw a ValueError.
            "ac" or "ca" - just return the existing one if it is a group. 
                           Otherwise throw a ValueError.
        '''
        if flag not in ('x', 'ac', 'ca'):
            raise ValueError(f'Invalid argument {flag=}')

        if key in self._raw.keys():
            if flag == 'x':
                raise ValueError(f'Redundant key {key}')
            val = self._raw[key]
            if not isinstance(val, h5py.Group):
                raise ValueError(f'key {key} does not refer to a group')
            val = Group(val)
        else:
            val = Group(self._raw.create_group(key))

        return val

    def create_soft_link(
            self, key: str, path: str, flag: CreateFlag = 'x') -> None:
        if flag not in ('x', 'ac', 'ca'):
            raise ValueError(f'Invalid argument {flag=}')

        if key in self:
            if flag == 'x':
                raise ValueError(f'Redundant key {key}')
        else:
            self._raw[key] = h5py.SoftLink(path)

    def create_external_link(self, key: str, file_name: str, path: str = '/',
                             flag: CreateFlag = 'x') -> None:
        if flag not in ('x', 'ac', 'ca'):
            raise ValueError(f'Invalid argument {flag=}')

        if key in self:
            if flag == 'x':
                raise ValueError(f'Redundant key {key}')
        else:
            self._raw[key] = h5py.ExternalLink(file_name, path)

    def dump(self, data_dict: Mapping, flag: DumpFlag = 'x') -> None:
        '''
        Dump `data_dict` as datasets and subgroups under this group.
        
        Supported value types: 
        - Mapping: recursively dump as a subgroup.
        - str, bytes, or np.ndarray: dump as a dataset.
        All the keys must be strings.
        
        The whole `data_dict` is scanned before the actual dumping. If any of 
        the key or value is not supported, a TypeError is raised and no dump 
        is made.
        
        Caution: this is not an atomic operation. For example, if a dataset is 
        failed to be created, the previous created ones will not be removed.
        
        @flag: applied to every dataset or group creation.
        '''
        chk_res, hint = Utils.is_data_dict_supported(data_dict)
        if not chk_res:
            raise TypeError(f'Unsupported data_dict: {hint}')
        dsets = self.datasets
        for k, v in data_dict.items():
            if isinstance(v, Mapping):
                self.create_group(k, flag=flag).dump(v, flag=flag)
            else:
                dsets.create(k, v, flag=flag)

    def load(self) -> DataDict:
        '''
        Load the data group as a DataDict.
        '''
        out = {}
        for k, v in self.items():
            if isinstance(v, Group):
                v_ld = v.load()
            elif isinstance(v, Dataset):
                v_ld = v[()]
            else:
                raise TypeError(f'Unknown value type {type(v)} for key {k}')
            out[k] = v_ld
        return DataDict(out)

    def what(self, attr=True, dset=True, group=True,
             max_depth=16, depth=0) -> str:
        out = '/'
        if attr:
            out += self.attrs.what()
        depth += 1
        if depth == max_depth:
            return out

        n_keys = len(self)
        pre = pre = '\n├─ '
        for i, (k, v) in enumerate(self.items()):
            if isinstance(v, Dataset):
                if not dset:
                    continue
                s = v.what(attr=attr)
            elif isinstance(v, Group):
                if not group:
                    continue
                s = v.what(attr=attr, dset=dset, group=group,
                           max_depth=max_depth, depth=depth)
            else:
                raise TypeError(f'Unknown value type {type(v)} for key {k}')

            if i == n_keys-1:
                pre = '\n└─ '

            out += pre + k + s.replace('\n', '\n   ')

        return out

    def keys(self) -> KeyList:
        return KeyList(self._raw.keys())

    def values(self) -> Iterator[NamedObj.Concrete]:
        for k in self._raw.keys():
            yield self[k]

    def items(self) -> Iterator[Tuple[str, NamedObj.Concrete]]:
        for k in self._raw.keys():
            yield k, self[k]

    @property
    def datasets(self) -> DatasetManager:
        return DatasetManager(self._raw)

    def __len__(self) -> int:
        return len(self._raw)

    def __getitem__(self, key: Utils.KeyOrKeys) \
            -> Union[NamedObj.Concrete, Tuple[NamedObj.Concrete, ...]]:

        if not isinstance(key, str):
            return tuple(self[k] for k in key)

        val = self._raw[key]
        if isinstance(val, h5py.Dataset):
            val = Dataset(val)
        elif isinstance(val, h5py.Group):
            val = Group(val)
        else:
            raise TypeError(f'Unknown value type {type(val)} for key {key}')

        return val

    def __contains__(self, key: str) -> bool:
        return key in self._raw

    def __delitem__(self, key: str) -> None:
        del self._raw[key]


class File(Group):

    Raw = h5py.File

    _flag_to_raw_flag = {
        'r':    'r',
        'a':    'r+',
        'x':    'x',
        'ac':   'a',
        'ca':   'a',
        'w':    'w',
    }

    OpenFlag = Literal['r', 'a', 'x', 'ac', 'ca', 'w']

    def __init__(
            self, path: Path = None,
            flag: OpenFlag = 'r') -> None:
        '''
        @flag: one of the following:
            - 'r': readonly mode for an existing file.
            - 'a': r/w mode for an existing file.
            - 'x': r/w mode for a newly-created file (fail if existing).
            - 'ca' | 'ac': r/w mode, open or create.
            - 'w': r/w mode, truncate or create.
        '''

        raw_flag = File._flag_to_raw_flag.get(flag)
        if raw_flag is None:
            raise ValueError(f'Invalid argument {flag=}')

        if path is None:
            h5_file = None
        else:
            h5_file = h5py.File(path, raw_flag)

        super().__init__(raw=h5_file)
        self._raw: File.Raw

    @staticmethod
    def ls_from(path: Path, key: str = '/', **ls_kw):
        with File(path) as f:
            f[key].ls(**ls_kw)
        
    @staticmethod
    def load_from(path: Path, key: str = None) \
            -> Union[DataDict, Utils.SupportedDataFromLoad]:
        '''
        Load a group or a dataset from a file.
        
        Examples
        --------
        p = 'file.hdf5'
        
        File.load_from(p)
            -> DataDict for the root data group.
        File.load_from(p, '/')
            -> the same as above.
        File.loadf_from(p, 'subgroup_a/subgroup_b') 
            -> DataDict for the group keyed subgroup_b under subgroup_a.
        File.loadf_from(p, 'subgroup_a/dataset_c')
            -> the dataset keyed dataset_c under subgroup_a.
        '''
        with File(path) as f:
            g = f if key is None else f[key]
            if isinstance(g, Group):
                out = g.load()
            elif isinstance(g, Dataset):
                out = g[()]
            else:
                raise TypeError(f'Unknown value type {type(g)} for key {key}')
        return out

    @staticmethod
    def dump_to(path: Path,
                data: Union[Mapping, Utils.SupportedData],
                key: str = None,
                f_flag: OpenFlag = 'ac',
                g_flag: Group.CreateFlag = 'x',
                dump_flag: Group.DumpFlag = 'x',
                create_parent=True) -> None:
        '''
        Dump a dict to a file, or create a dataset to hold `data`.
        
        @key: for a dict, it is dumped as datasets or sub groups under 
            file[key], or the root group if key is None. For a single piece 
            of data (e.g., np.ndarray, str), it is dumped so that file[key] 
            should return the dataset holding it.
        @f_flag: flag passed to h5.File().
        @g_flag: significant only when `key` is not None and a group should 
            be created to hold the dict. The parent groups must exist if 
            create_parent is False, or created on needing if True.
        @dump_flag: passed to Group.dump() for a dict, and passed to 
            DatasetManager.create() for a single piece of data.
        
        Examples
        --------
        p = 'file.hdf5'
        d = {'a': 1, 'b': 2, 'c': {'d': 3}}
        
        File.dump_to(p, d)
            -> dump as datasets/data groups under the root group.
        File.dump_to(p, d, key='/')
            -> the same as above.
        
        File.dump_to(p, d, key='c/d')
            -> dump to the group keyed 'd' under the group keyed 'c'. 'c' is
               opened, or created if not existing. 'd' is created.
        File.dump_to(p, d, key='c/d', g_flag='ac', dump_flag = 'ac')
            -> the same, but 'd' is opened if existing, and the data under 
               it are overwritten for redundant names appearing in `data`.
        
        File.dump_to(p, d['b'], key='c/d')
            -> dump as a dataset 'd' under 'c'. 'c' is opened, or created if
               not existing.
               
        particles = {
            'pos': np.random.rand(100, 3),
            'vel': np.random.rand(100, 3),
            'id': np.arange(100),
            'n_particles: 100,
        }
        File.dump_to(p, particles, key='Snapshots/99')
            -> a real astronomical application.
        '''
        with File(path, flag=f_flag) as f:
            if key is None:
                f.dump(data, flag=dump_flag)
                return
            ks = [k for k in key.split('/') if len(k) > 0]
            if len(ks) == 0:    # e.g., key = '/'
                f.dump(data, flag=dump_flag)
                return
            g = f
            for k in ks[:-1]:
                if create_parent:
                    g = g.create_group(k, flag='ac')
                else:
                    g = g[k]
                    if not isinstance(g, Group):
                        raise ValueError(f'key {k} does not refer to a group')
            if isinstance(data, Mapping):
                g.create_group(ks[-1], flag=g_flag).dump(data, flag=dump_flag)
            else:
                g.datasets.create(ks[-1], data, flag=dump_flag)

    @staticmethod
    def from_raw(raw: Raw = None) -> File:
        f = File()
        f._raw = raw
        return f

    def close(self) -> None:
        if self._raw is not None:
            self._raw.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self._raw is not None:
            self._raw.__exit__(*args)
