from __future__ import annotations
import json
from pyhipp.core import DataDict
import numpy as np
from pathlib import Path

class File:
    
    _flag_to_raw_flag = {
        'r':    'r',
        'a':    'r+',
        'x':    'x',
        'ac':   'a',
        'ca':   'a',
        'w':    'w',
    }
    
    @staticmethod
    def load_file(path: Path, parse=True, **load_kw):
        '''
        @parse: whether to parse the loaded data using File.parse_input.
        '''
        with open(path) as f:
            out = json.load(f, **load_kw)
        if parse:
            out = File.parse_input(out)
        return out

    @staticmethod
    def dump_file(obj, path: Path, flag='x', use_default=True, **dump_kw):
        flag = File._flag_to_raw_flag[flag]
        if use_default:
            dump_kw['default'] = File.default_dump
        with open(path, flag) as f:
            json.dump(obj, f, **dump_kw)

    @staticmethod
    def parse_input(d):
        '''
        Parse the loaded data from JSON file. The mapping is as follows:
        - dict -> DataDict
        - list -> np.ndarray
        - str, int, float -> themselves
        - otherwise -> themselves
        '''
        if isinstance(d, dict):
            return DataDict({File.parse_input(k): File.parse_input(v)
                            for k, v in d.items()})
        if isinstance(d, (str, int, float)):
            return d
        if isinstance(d, list):
            return np.array(d)
        return d

    @staticmethod
    def default_dump(obj):
        if isinstance(
                obj, (np.ndarray, np.int64, np.int32, np.float64, np.float32)):
            return obj.tolist()
        elif isinstance(obj, DataDict):
            return obj._dict
        raise TypeError(f'Object of type {type(obj)} is not JSON serializable')

    @staticmethod
    def load_table(path: Path, key: str | list[str] | None = 'table'):
        table = File.load_file(path, parse=False)
        if key is not None:
            if isinstance(key, str):
                table = table[key]
            else:
                for k in key:
                    table = table[k]
        return File.parse_table(table)

    @staticmethod
    def parse_table(table: dict) -> DataDict[str, DataDict[str, str] |
                                             DataDict[str, np.ndarray]]:
        cols = table['header']['columns']
        data = table['data']
        out_data = DataDict()
        out_descr = DataDict()
        for i_c, (key, dtype, descr) in enumerate(cols):
            vals = np.array([row[i_c] for row in data], dtype=dtype)
            out_data[key] = vals
            out_descr[key] = descr
        return DataDict({
            'data': out_data,
            'descr': out_descr,
        })
