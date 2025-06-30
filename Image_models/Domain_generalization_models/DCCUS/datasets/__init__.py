from __future__ import absolute_import
from .Crowd import CrowdTest
from .CrowdDataset import CrowdCluster

__factory = {'Crowd': CrowdTest, 'CrowdCluster': CrowdCluster}

def create(name, root, *args, **kwargs):
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root, *args, **kwargs)