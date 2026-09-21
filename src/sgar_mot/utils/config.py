"""YAML configuration loading and system utilities."""

import copy
import os
import resource
import types

import yaml


def load_args_from_config(config_file):
    """Loads a MIR configuration file.

    The file has a ``transformer`` section, which defines the architecture and
    the weights to load, and an optional ``trainer`` section, only needed when
    training. Returns a namespace with one attribute per section.
    """
    with open(config_file, "r") as stream:
        args = yaml.safe_load(stream)

    config = types.SimpleNamespace()
    config.transformer = types.SimpleNamespace(**args["transformer"])

    if "trainer" in args:
        config.trainer = types.SimpleNamespace(**args["trainer"])
        config.trainer.transformer = config.transformer

    return config


def merge_args(base_args, new_args, verbose=True):
    """Overrides the values of ``base_args`` with the ones given in ``new_args``."""
    base_args = copy.deepcopy(base_args)
    for key, value in new_args.__dict__.items():
        if key in base_args.__dict__ and value is not None:
            if verbose:
                print('Overriding {} from {} to {}'.format(key, base_args.__dict__[key], value), flush=True)
            setattr(base_args, key, value)

        elif key not in base_args.__dict__:
            setattr(base_args, key, value)
            if verbose:
                print('Setting {} to {}'.format(key, value), flush=True)

    return base_args


def get_ram_usage():
    process = resource.getrusage(resource.RUSAGE_SELF)
    return process.ru_maxrss * 1024


def get_total_ram():
    return os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
