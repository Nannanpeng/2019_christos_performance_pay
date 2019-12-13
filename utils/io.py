import numpy as np
import pandas as pd
import yaml
from contextlib import contextmanager
import logging
logger = logging.getLogger(__name__)
import ctypes
import io
import os, sys
import tempfile
import pickle
from collections import namedtuple

__all__ = [
    'struct_factory', 'yaml_to_model', 'load_yaml', 'stdout_redirector',
    'save_checkpoint', 'load_checkpoint'
]


def struct_factory(name, dictionary):
    return namedtuple(name, dictionary.keys())(**dictionary)


# model = namedtuple('ModelSpec',['constants','parameters','dynamics'])


def load_checkpoint(path):
    logger.info('Input file: %s' % path)
    V = None
    with open(path, 'rb') as fd_old:
        V = pickle.load(fd_old)
        logger.info("Saved checkpoint at path %s loaded from disk" % path)
    return V


def save_checkpoint(V, path):
    logger.info('Output file: %s' % path)
    with open(path, 'wb') as fd:
        pickle.dump(V, fd, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved checkpoint written to disk at path: %s" % path)


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
        parameters = struct_factory('%s_params' % name, parameters)
    return {
        'constants': constants,
        'name': name,
        'parameters': parameters,
        'dynamics':
        model_dict['dynamics'] if 'dynamics' in model_dict else None
    }


def load_yaml(file_path):
    with open(file_path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


libc = ctypes.CDLL(None)
c_stdout = ctypes.c_void_p.in_dll(libc, 'stdout')


# Source: https://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/
@contextmanager
def stdout_redirector(stream):
    # The original fd stdout points to. Usually 1 on POSIX systems.
    original_stdout_fd = sys.stdout.fileno()

    def _redirect_stdout(to_fd):
        """Redirect stdout to the given file descriptor."""
        # Flush the C-level buffer stdout
        libc.fflush(c_stdout)
        # Flush and close sys.stdout - also closes the file descriptor (fd)
        sys.stdout.close()
        # Make original_stdout_fd point to the same file as to_fd
        os.dup2(to_fd, original_stdout_fd)
        # Create a new sys.stdout that points to the redirected fd
        sys.stdout = io.TextIOWrapper(os.fdopen(original_stdout_fd, 'wb'))

    # Save a copy of the original stdout fd in saved_stdout_fd
    saved_stdout_fd = os.dup(original_stdout_fd)
    try:
        # Create a temporary file and redirect stdout to it
        tfile = tempfile.TemporaryFile(mode='w+b')
        _redirect_stdout(tfile.fileno())
        # Yield to caller, then redirect stdout back to the saved fd
        yield
        _redirect_stdout(saved_stdout_fd)
        # Copy contents of temporary file to the given stream
        tfile.flush()
        tfile.seek(0, io.SEEK_SET)
        stream.write(tfile.read())
    finally:
        tfile.close()
        os.close(saved_stdout_fd)