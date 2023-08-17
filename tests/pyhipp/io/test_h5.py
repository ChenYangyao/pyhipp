import pytest
from pathlib import Path
from pyhipp.io import h5
from pyhipp.core import DataDict
import numpy as np

@pytest.fixture
def file_1(tmp_path: Path):
    p = tmp_path / 'file_1.hdf5'
    f = h5.File(p, 'w')
    yield DataDict({'f': f, 'path': p})
    f.close()

@pytest.fixture
def file_2(tmp_path: Path):
    p = tmp_path / 'file_2.hdf5'
    f = h5.File(p, 'w')
    yield DataDict({'f': f, 'path': p})
    f.close()

def create_group(f: h5.File):
    g_a = f.create_group("g_a")
    g_b = f.create_group("g_b")
    g_ac = g_a.create_group("g_ac")
    g_acd = g_ac.create_group("g_acd")
    
    assert 'g_a' in f.keys()
    assert 'g_b' in f.keys()
    assert 'g_ac' in g_a.keys()
    assert 'g_acd' in g_ac.keys()
    
    assert isinstance(f['g_a'], h5.Group)
    assert isinstance(f['g_b'], h5.Group)
    assert isinstance(f['g_a/g_ac'], h5.Group) 
    assert isinstance(f['g_a/g_ac/g_acd'], h5.Group) 
    
    attrs = g_a.attrs
    attrs.create('attr1', np.array(1))
    attrs.create('attr2', 'qq')
    attrs.create('attr3', np.array([1,2,3]))
    
    return DataDict({
        'f': f,
        'g_a': g_a, 'g_b': g_b, 'g_ac': g_ac, 'g_acd': g_acd,
    })
    
    
@pytest.fixture
def grp_1(file_1: DataDict):
    return create_group(file_1['f'])

def test_create_group(grp_1):
    assert isinstance(grp_1, DataDict)


def create_dataset(g: h5.Group):
    a = np.arange(6, dtype=int)
    b = np.ones((10,10), dtype=int)
    c = 1
    d = b'121c'
    e = '121c'
    
    dsets = g.datasets
    d_a = dsets.create('d_a', a)
    d_b = dsets.create('d_b', b)
    d_c = dsets.create('d_c', c)
    d_d = dsets.create('d_d', d)
    d_e = dsets.create('d_e', e)
    
    with pytest.raises(ValueError):
        dsets.create('d_b', b)
    with pytest.raises(ValueError):
        dsets.create('d_b', b, flag='wrong flag')

    b += 1    
    d_b = dsets.create('d_b', b + 1, flag='ac')
    
    b+= 2
    d_b = dsets.create('d_b', b, flag='ca')

    assert (d_b[()] == b).all()
    
    attrs = d_c.attrs
    attrs.create('name', 'd_c')
    attrs.create('values', np.array([1,2,3,4,5.0]))
    
    dsets = g['g_ac/g_acd'].datasets
    d_acd_a = dsets.create('d_acd_a', '111')
    d_acd_b = dsets.create('d_acd_b', np.empty((3,4,5)))
    d_acd_b.attrs.create('33', np.ones((3,3)))

    return DataDict({
        'g': g,
        'd_a': d_a, 'd_b': d_b, 'd_c': d_c, 'd_d': d_d, 'd_e': d_e,
        'd_acd_a': d_acd_a, 'd_acd_b': d_acd_b,
        'data': DataDict({
            'a': a, 'b': b, 'c': c, 'd': d, 'e': e,
        }),
    })

@pytest.fixture
def dset_1(grp_1: DataDict):
    return create_dataset(grp_1['g_a'])


def test_create_dataset(dset_1):
    assert isinstance(dset_1, DataDict)

def test_create_soft_link(grp_1):
    g_b: h5.Group = grp_1['g_b']
    g_b2 =g_b.create_group('g_b1').create_group('g_b2')
    g_b.create_soft_link('link_to_g_b2', 'g_b1/g_b2')

    g_b2.datasets.create('d', np.arange(5))
    assert isinstance(g_b['link_to_g_b2'], h5.Group)
    assert isinstance(g_b['link_to_g_b2/d'], h5.Dataset)
    assert np.all(g_b['link_to_g_b2/d'][()] == np.arange(5))

def test_create_external_link(file_1, file_2):
    p_1: Path = file_1['path']
    f_1: h5.File = file_1['f']
    f_2: h5.File = file_2['f']
    f_1.create_group('g_1').datasets.create('d', np.arange(5))
    f_2.create_group('g_2').create_external_link(
        'link_to_g_1', str(p_1), '/g_1')
    assert np.all(f_2['g_2/link_to_g_1/d'][()] == np.arange(5))

def test_load_multiple_keys(grp_1, dset_1):
    g_a: h5.Group = grp_1['g_a']
    a,b,c,d,e = dset_1['data']['a', 'b', 'c', 'd', 'e']
    _a, _b, _c, _d, _e = g_a.datasets['d_a', 'd_b', 'd_c', 'd_d', 'd_e']
    assert (a == _a).all()
    assert (b == _b).all()
    assert c == _c
    assert d == _d
    assert e.encode() == _e
    print(_a, _b, _c, _d, _e)
    
def test_ls(file_1, grp_1, dset_1):
    f: h5.File = file_1['f']
    f.ls()
    g_a: h5.Group = grp_1['g_a']
    g_a.ls()
    
    f.ls(max_depth=1)
    f.ls(max_depth=2)
    f.ls(max_depth=3)
    f.ls(max_depth=4)
    f.ls(max_depth=5)

def test_dataset_manager_dump(grp_1):
    g_b: h5.Group = grp_1['g_b']
    dsets = g_b.datasets
    assert isinstance(dsets, h5.DatasetManager)
    
    dsets.dump({
        'a': np.array([1,2,3], dtype=float),
        'b': 1,
        'c': np.array([[1,2],[3,4]], dtype=float),
    })
    g_b.create_group('d').datasets.dump({
        'd1': np.arange(5),
        'd2': '123',
        'd3': b'456',
    })
    
    g_b_ld: h5.Group = grp_1['f/g_b']
    for key in 'a', 'b', 'c', 'd':
        assert key in g_b_ld._raw
    assert (g_b_ld['a'][()] == [1,2,3]).all()
    g_b_d_ld: h5.Group = g_b_ld['d']
    for key in 'd1', 'd2', 'd3':
        assert key in g_b_d_ld._raw
    assert (g_b_d_ld['d1'][()] == [0,1,2,3,4]).all()

def test_dataset_manager_load(grp_1):
    test_dataset_manager_dump(grp_1)
    g_b: h5.Group = grp_1['g_b']
    d = g_b.datasets.load()
    assert len(d) == 3
    assert (d['a'] == np.array([1,2,3])).all()
    assert d['b'] == 1
    assert (d['c'] == np.array([[1,2],[3,4]])).all()

def test_group_dump(grp_1):
    g_b: h5.Group = grp_1['g_b']
    g_b.dump({
        'a': np.array([1,2,3], dtype=float),
        'b': 1,
        'c': np.array([[1,2],[3,4]], dtype=float),
        'd': {
            'd1': np.arange(5),
            'd2': '123',
            'd3': b'456',
        }
    })
    
    g_b_ld: h5.Group = grp_1['f/g_b']
    for key in 'a', 'b', 'c', 'd':
        assert key in g_b_ld._raw
    assert (g_b_ld['a'][()] == [1,2,3]).all()
    g_b_d_ld: h5.Group = g_b_ld['d']
    for key in 'd1', 'd2', 'd3':
        assert key in g_b_d_ld._raw
    assert (g_b_d_ld['d1'][()] == [0,1,2,3,4]).all()
    
def test_group_load(grp_1):
    test_group_dump(grp_1)
    g_b: h5.Group = grp_1['g_b']
    d = g_b.load()
    assert len(d) == 4
    assert (d['a'] == np.array([1,2,3])).all()
    assert d['b'] == 1
    assert (d['c'] == np.array([[1,2],[3,4]])).all()

    d = d['d']
    assert (d['d1'] == np.arange(5)).all()
    assert d['d2'] == b'123'
    assert d['d3'] == b'456'

def test_file_dump_to(tmp_path: Path):
    d = {
        'a': np.array([1,2,3], dtype=float),
        'b': 1,
        'c': np.array([[1,2],[3,4]], dtype=float),
        'd': {
            'd1': np.arange(5),
            'd2': '123',
            'd3': b'456',
        }
    }
    file_name = tmp_path / 'f0.hdf5'
    h5.File.dump_to(file_name, d)
    with h5.File(file_name) as f:
        d_ld = f['/'].load()
    assert len(d) == len(d_ld)
    for k in 'a', 'b', 'c':
        assert np.all(d[k] == d_ld[k])
    assert len(d['d']) == len(d_ld['d'])
    for k in 'd1', 'd2', 'd3':
        v = d['d'][k]
        if isinstance(v, str):
            v = v.encode()
        assert np.all(v == d_ld[f'd/{k}'])

def test_file_dump_load_example(tmp_path: Path):
    
    halo_cat = {
        'Header': {
            'n_subhalos': 10, 'n_halos': 5, 'version': '1.0.0',
            'source': 'ELUCID simulation', 'last_update': '2023-08-17',
        },
        'Subhalos': {
            'id': np.arange(10),
            'x': np.random.uniform(size=(10,3)), 
            'v': np.random.uniform(size=(10,3)),
        },
        'Halos': {
            'id': np.arange(5),
            'x': np.random.uniform(size=(5,3)),
            'v': np.random.uniform(size=(5,3)),
        },
    }
    
    '''Dump all data recursively into a HDF5 file.'''
    path = tmp_path / 'halo_cat.hdf5'
    h5.File.dump_to(path, halo_cat)
    
    '''Load back all, or a subset of data.'''
    halo_cat = h5.File.load_from(path)
    halos = h5.File.load_from(path, 'Halos')
    
    print(halo_cat)
    print(halos)
    
    '''Of course, you can open the file, and load datasets separately.'''
    with h5.File(path) as f:
        dsets = f['Halos'].datasets
        x = dsets.x                   # load via attributes (Thanks Zhaozhou Li for the idea)
        id, v = dsets['id', 'v']      # load via getitem
        
        halos = f['Halos'].load()     # load all halos as a dict-like object
        x = halos['x']
    
    print(id, x, v)
    