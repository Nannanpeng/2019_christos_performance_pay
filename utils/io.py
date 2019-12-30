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
    'struct_factory', 'yaml_to_spec', 'load_yaml', 'stdout_redirector',
    'stderr_redirector', 'save_model', 'load_model', 'ModelSpec',
    'ipopt_stdout_filter'
]

ModelSpec = namedtuple('ModelSpec',
                       ['constants', 'parameters', 'dynamics', 'name'])
RunSpec = namedtuple('RunSpec', ['model', 'algorithm_config'])


def struct_factory(name, dictionary):
    return namedtuple(name, dictionary.keys())(**dictionary)


def load_model(path):
    logger.info('Input file: %s' % path)
    V = None
    with open(path, 'rb') as fd_old:
        V = pickle.load(fd_old)
        logger.info("Saved model at path %s loaded from disk" % path)
    return V


def save_model(V, path):
    logger.info('Output file: %s' % path)
    with open(path, 'wb') as fd:
        pickle.dump(V, fd, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved model written to disk at path: %s" % path)


def _load_value(kind, val):
    if kind == 'str':
        return str(val)
    if kind == 'float':
        return float(val)
    if kind == 'int':
        return int(val)
    if kind == 'array':
        return np.array(val)
    if kind == 'csv':
        return pd.read_csv(val)


def _process_entry(entry, entry_name, model_name):
    entry_val = {
        k: _load_value(v['kind'], v['value'])
        for (k, v) in entry.items()
    }
    return struct_factory('%s_%s' % (model_name, entry_name), entry_val)


def yaml_to_spec(model_dict):
    name = model_dict['name']

    constants = _process_entry(model_dict['constants'], 'constants',
                               name) if 'constants' in model_dict else None
    parameters = _process_entry(model_dict['parameters'], 'parameters',
                                name) if 'parameters' in model_dict else None
    algorithm = _process_entry(
        model_dict['algorithmConfig'], 'algorithmConfig',
        name) if 'algorithmConfig' in model_dict else None
    dynamics = model_dict['dynamics'] if 'dynamics' in model_dict else None
    return RunSpec(model=ModelSpec(constants=constants,
                                   parameters=parameters,
                                   dynamics=dynamics,
                                   name=name),
                   algorithm_config=algorithm)


def load_yaml(file_path):
    with open(file_path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def ipopt_stdout_filter(val, _logger):
    if val.strip() == '':
        return
    if '[IPyOpt]' in val:
        return
    logger.debug(val)


libc = ctypes.CDLL(None)
c_stdout = ctypes.c_void_p.in_dll(libc, 'stdout')
c_stderr = ctypes.c_void_p.in_dll(libc, 'stderr')

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


@contextmanager
def stderr_redirector(stream):
    # The original fd stderr points to. Usually 1 on POSIX systems.
    original_stderr_fd = sys.stderr.fileno()

    def _redirect_stderr(to_fd):
        """Redirect stderr to the given file descriptor."""
        # Flush the C-level buffer stderr
        libc.fflush(c_stderr)
        # Flush and close sys.stderr - also closes the file descriptor (fd)
        sys.stderr.close()
        # Make original_stderr_fd point to the same file as to_fd
        os.dup2(to_fd, original_stderr_fd)
        # Create a new sys.stderr that points to the redirected fd
        sys.stderr = io.TextIOWrapper(os.fdopen(original_stderr_fd, 'wb'))

    # Save a copy of the original stderr fd in saved_stderr_fd
    saved_stderr_fd = os.dup(original_stderr_fd)
    try:
        # Create a temporary file and redirect stderr to it
        tfile = tempfile.TemporaryFile(mode='w+b')
        _redirect_stderr(tfile.fileno())
        # Yield to caller, then redirect stderr back to the saved fd
        yield
        _redirect_stderr(saved_stderr_fd)
        # Copy contents of temporary file to the given stream
        tfile.flush()
        tfile.seek(0, io.SEEK_SET)
        stream.write(tfile.read())
    finally:
        tfile.close()
        os.close(saved_stderr_fd)
