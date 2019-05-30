import numpy as np
from collections import namedtuple

import utils

ModelSpec = namedtuple("ModelSpec",["a"])





if __name__ == "__main__":
    utils.load_model_from_yaml('./example.yaml')
