from __future__ import annotations
import typing
from typing import Self
from pathlib import Path
from .abc import HasDictRepr
import os


class FileSystemPath(HasDictRepr):

    repr_attr_keys = ('path',)

    def __init__(self, path: Path) -> None:
        if isinstance(path, FileSystemPath):
            path = path.path
        self.path = Path(path)

    def __str__(self):
        return str(self.path)

    def to_str(self) -> str:
        return str(self)

    def exists(self) -> bool:
        return self.path.exists()

    def is_file(self) -> bool:
        return self.path.is_file()

    def is_dir(self) -> bool:
        return self.path.is_dir()

    def __truediv__(self, other):
        return FileSystemPath(self.path / other)


class File(FileSystemPath):
    def load_json(self, parse=True, **load_kw):
        from ..io.json import File
        return File.load_file(self.path, parse=parse, **load_kw)

    def load_h5(self, key: str | None = None, **load_kw):
        from ..io.h5 import File
        return File.load_from(self.path, key=key, **load_kw)

    def open_h5(self, flag='r', **open_kw):
        from ..io.h5 import File
        return File(self.path, flag=flag, **open_kw)

    def dump_json(self, data, flag='x', use_default=True, **dump_kw):
        from ..io.json import File
        File.dump_file(data, self.path, flag=flag, use_default=use_default,
                       **dump_kw)
        return self

    def dump_h5(self, data, key: str | None = None,
                f_flag='x',
                g_flag='x',
                dump_flag='x',
                **dump_kw):
        from ..io.h5 import File
        File.dump_to(self.path, data, key=key, f_flag=f_flag,
                     g_flag=g_flag, dump_flag=dump_flag, **dump_kw)
        return self

    def load(self, **load_kw):
        if self.path.suffix == '.json':
            out = self.load_json(**load_kw)
        elif self.path.suffix in ('.h5', '.hdf5'):
            out = self.load_h5(**load_kw)
        else:
            raise ValueError(f'Unsupported suffix {self.path.suffix}'
                             f' in file {self.path}')
        return out

    def open(self, **open_kw):
        assert self.path.suffix in ('.h5', '.hdf5')
        return self.open_h5(**open_kw)

    def dump(self, data, **dump_kw):
        if self.path.suffix == '.json':
            out = self.dump_json(data, **dump_kw)
        elif self.path.suffix in ('.h5', '.hdf5'):
            out = self.dump_h5(data, **dump_kw)
        else:
            raise ValueError(f'Unsupported suffix {self.path.suffix}'
                             f' in file {self.path}')
        return out

    def save_fig(self, **savefig_kw):
        from ..plot import savefig
        savefig(self.path, **savefig_kw)


class Directory(FileSystemPath):
    def subdir(self, name: str):
        return Directory(self / name)

    def file(self, name: str):
        return File(self / name)

    def mkdir(self, mode=0o755, exist_ok=False, parents=False):
        self.path.mkdir(mode=mode, exist_ok=exist_ok, parents=parents)
        return self


class BaseProject(HasDictRepr):
    
    repr_attr_keys = ('project_dir', 'data_dir', 'figs_dir', 'sims_dir',
                      'obss_dir', 'tables_dir', 'products_dir')
    
    parent_dir: Path | FileSystemPath = Path(
        os.environ.get('MAHGIC_WORK_DIR', os.getcwd())
    ).resolve()
    project_name: str = 'PyHippProject'
    subproject_names: tuple[str] = ()

    data_dir_name = 'data'

    figs_dir_name = 'figs'
    sims_dir_name = 'sims'
    obss_dir_name = 'obss'
    tables_dir_name = 'tables'
    products_dir_name = 'products'

    def __init__(self):
        _p = self.parent_dir / self.project_name
        for name in self.subproject_names:
            _p = _p / name

        project_dir = Directory(_p)
        data_dir = project_dir.subdir(self.data_dir_name)

        self.project_dir = project_dir
        self.data_dir = data_dir

        self.figs_dir = data_dir.subdir(self.figs_dir_name)
        self.sims_dir = data_dir.subdir(self.sims_dir_name)
        self.obss_dir = data_dir.subdir(self.obss_dir_name)
        self.tables_dir = data_dir.subdir(self.tables_dir_name)
        self.products_dir = data_dir.subdir(self.products_dir_name)

    def make_filesystem(self, exist_ok=False, parents=False):
        assert self.project_dir.exists()
        for d in (self.data_dir,
                  self.figs_dir, self.sims_dir, self.obss_dir,
                  self.tables_dir, self.products_dir):
            d.mkdir(exist_ok=exist_ok, parents=parents)
