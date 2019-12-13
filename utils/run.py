# Utility Functions
# Mainly for configuration management

import argparse
import os
import time
from .io import yaml_to_spec, load_yaml

__all__ = ['make_directory', 'run_config_from_args', 'gpr_argparser']


def make_directory(dirpath):
    os.makedirs(dirpath, exist_ok=True)

def _resolve_attr(spec,args,attrname, conversion = None):
    val = getattr(spec,attrname, None)
    val = val if val is not None else getattr(args,attrname,None)
    if conversion is not None and val is not None:
        return conversion(val)
    return val


def run_config_from_args(args):
    spec = yaml_to_spec(load_yaml(args.spec))
    odir_alt = 'out/%s_%s' % (spec.model.name, time.strftime("%Y.%m.%d_%H.%M.%S"))
    run_config = {
        'debug': args.debug,
        'terminal': args.terminal,
        'odir': args.odir if args.odir is not None else odir_alt,
        'model': spec.model,
        'seed': _resolve_attr(spec.algorithm_config,args,'seed',int),
        'tolerance': _resolve_attr(spec.algorithm_config,args,'tolerance',int),
        'max_updates': _resolve_attr(spec.algorithm_config,args,'max_updates',int),
        'save_interval': _resolve_attr(spec.algorithm_config,args,'save_interval',int),
        'algorithm_config': spec.algorithm_config
    }

    return run_config


def gpr_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o",
                        "--odir",
                        type=str,
                        default=None,
                        help="output directory")
    parser.add_argument("-d", "--debug", action="store_true", help="debug")
    parser.add_argument("-u",
                        '--max_updates',
                        type=int,
                        default=100,
                        help="max number of iterations")
    parser.add_argument("-e",
                        '--tolerance',
                        type=float,
                        default=1e-4,
                        help="error tolerance for stopping condition")
    parser.add_argument("-c",
                        '--checkpoint_interval',
                        type=float,
                        default=1,
                        help="interval at which to save checkpoint")
    parser.add_argument("-s",
                        '--spec',
                        type=str,
                        default="./model_specs/simon_gpr.yaml",
                        help="What model / spec to load")
    parser.add_argument("-t",
                        "--terminal",
                        action="store_true",
                        help="log to console")
    parser.add_argument(
        '--seed',
        type=int,
        default=543,
        metavar='N',
        help='random seed (default: 543). -1 indicates no seed')

    return parser
