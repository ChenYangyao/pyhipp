from __future__ import annotations
from pathlib import Path
from typing import List
from ..core import DataDict, abc
import yaml

class ConfigFileParser(abc.HasLog):
    def __init__(self, file_name: Path, init_nodes: List[str], 
                 nested_nodes: List[str],
                 verbose = False) -> None:
        super().__init__(verbose=verbose)
        
        self.file_name = Path(file_name)
        self.init_nodes = list(init_nodes)
        self.nested_nodes = list(nested_nodes)
        
    def get(self, keys: List[str]) -> DataDict:
        with open(self.file_name, 'r') as f:
            data = yaml.load(f, Loader=yaml.Loader) 
        self.log(f'Loaded from file {self.file_name}')
        
        for k in self.init_nodes:
            data = data[k]
        self.log(f'Init nodes = {self.init_nodes}')
            
        d = DataDict()
        self.__try_parse(keys, d, data)
        for k in self.nested_nodes:
            data = data[k]
            self.__try_parse(keys, d, data)
        self.log(f'Nested nodes = {self.nested_nodes}')
        
        for k in keys:
            if k not in d:
                raise KeyError(f'Key {k} not found')
        self.log(f'Loaded config dict {d}')

        return d
        
    def __try_parse(self, keys: List[str], d: DataDict, data: dict):
        for k in keys:
            if k in data:
                d[k] = data[k]
            
        