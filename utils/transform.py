import numpy as np
import pandas as pd
from collections import namedtuple


__all__ = ['struct_factory','yaml_to_model']

def struct_factory(name,dictionary):
    return namedtuple(name, dictionary.keys())(**dictionary)

model = namedtuple('ModelSpec',['constants','parameters','dynamics'])

def _load_value(kind,val):
    if kind == 'scalar':
        return float(val)
    if kind == 'array':
        return np.array(val)
    if kind == 'csv':
        return pd.read_csv(val)

def yaml_to_model(model_dict):
    constants = { k: _load_value(v['kind'],v['value']) for (k,v) in model_dict['constants'].items() }
    parameters = { k: _load_value(v['kind'],v['value']) for (k,v) in model_dict['parameters'].items() }
    dynamics = model_dict['dynamics']

    return model(constants,parameters,dynamics)
