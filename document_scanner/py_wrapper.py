import importlib
import os
import sys

from itertools import groupby
from operator import itemgetter


def identity(x):
    return x


def compose(*functions):
    def inner(arg):
        for f in reversed(functions):
            arg = f(arg)
        return arg
    return inner


def aggregate_keys(dictionary, aggregator=set, key_wrapper=identity, value_wrapper=identity):
    return {value: aggregator(key_wrapper(key) for key, _ in group) for value, group in groupby(sorted(dictionary.items(), key=itemgetter(1)), key=compose(value_wrapper, itemgetter(1)))}


def aggregate_values(dictionary, aggregator=set, key_wrapper=identity, value_wrapper=identity):
    return {key: aggregator(value_wrapper(value) for _, value in group) for key, group in groupby(sorted(dictionary.items()), key=compose(key_wrapper, itemgetter(0)))}


def get_class_from_module_path(path):

    folder_path, filename = os.path.split(path)
    module_name = os.path.splitext(filename)[0]
    class_name = module_name.capitalize()

    # more targeted way of loading module that avoids adding folder_path to the system path,
    # but then how does one import sibling modules in module_at_path?
    # spec = importlib.util.spec_from_file_location(module_name, path)
    # module_at_path = importlib.util.module_from_spec(spec)
    # spec.loader.exec_module(module_at_path)

    sys.path.append(folder_path)
    module_at_path = importlib.import_module(module_name)
    class_at_path = getattr(module_at_path, class_name)

    return class_at_path
