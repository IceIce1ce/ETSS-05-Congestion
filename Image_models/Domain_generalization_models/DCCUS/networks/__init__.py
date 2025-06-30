from __future__ import absolute_import
from .MetaMemNet import *

__factory = {'memMeta': MetaMemNet}

def names():
    return sorted(__factory.keys())

def create(name, *args, **kwargs):
    if name not in __factory:
        raise KeyError("Unknown model:", name)
    return __factory[name](*args, **kwargs)