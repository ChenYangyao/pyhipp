import importlib_resources
import json

def test_resources():
    data_dir = importlib_resources.files('pyhipp.astro').joinpath('data/cosmologies')
    with importlib_resources.as_file(data_dir) as p:
        print(f'Locate resource dir {p}')
        with open(p / 'tng.json', 'rb') as f:
            cosm: dict = json.load(f)
        print(cosm)
            
        