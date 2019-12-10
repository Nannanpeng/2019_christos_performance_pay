import numpy as np
import pandas as pd
import yaml
from collections import namedtuple

__all__ = ['struct_factory', 'yaml_to_model', 'load_yaml']


def struct_factory(name, dictionary):
    return namedtuple(name, dictionary.keys())(**dictionary)


# model = namedtuple('ModelSpec',['constants','parameters','dynamics'])


def _load_value(kind, val):
    if kind == 'float':
        return float(val)
    if kind == 'int':
        return int(val)
    if kind == 'array':
        return np.array(val)
    if kind == 'csv':
        return pd.read_csv(val)


def yaml_to_model(model_dict):
    constants, parameters = None, None
    name = model_dict['name']
    if 'constants' in model_dict:
        constants = {
            k: _load_value(v['kind'], v['value'])
            for (k, v) in model_dict['constants'].items()
        }
        constants = struct_factory('%s_constants' % name, constants)
    if 'parameters' in model_dict:
        parameters = {
            k: _load_value(v['kind'], v['value'])
            for (k, v) in model_dict['parameters'].items()
        }
        parameters = struct_factory('%s_params' % name,parameters)
    return {
        'constants': constants,
        'name': model_dict['name'],
        'parameters': parameters,
        'dynamics': model_dict['dynamics'] if 'dynamics' in model_dict else None
    }


def load_yaml(file_path):
    with open(file_path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


class sample_token:
    def __init__(self, **response):
        for k, v in response.items():
            if isinstance(v, dict):
                self.__dict__[k] = sample_token(**v)
            else:
                self.__dict__[k] = v