from collections import namedtuple

__all__ = ['struct_factory']

def struct_factory(name,dictionary):
    return namedtuple(name, dictionary.keys())(**dictionary)
