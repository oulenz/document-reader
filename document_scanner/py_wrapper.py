import importlib
import os
import sys
import time

from functools import wraps
from itertools import groupby
from operator import itemgetter
from types import MethodType


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


def is_instance_method(cls, attribute):
    return hasattr(attribute, '__self__') and isinstance(getattr(attribute, '__self__'), cls)


def decorate_all_instance_methods(method_decorator):
    def class_decorator(cls):
        orig_init = cls.__init__
        def __init__(self, *args, **kwargs):
            self._method_times = []
            orig_init(self, *args, **kwargs)
        cls.__init__ = __init__
        
        orig_getattribute = cls.__getattribute__
        def __getattribute__(self, s):
            x = orig_getattribute(self, s)
            if is_instance_method(cls, x):
                x = MethodType(method_decorator(x.__func__), x.__self__)
            return x
        cls.__getattribute__ = __getattribute__
        
        return cls
    return class_decorator


def store_time(m):
    @wraps(m)
    def m_timed(self, *args, **kwargs):
        if not hasattr(self, '_method_times'):
            self._method_times = []
        start_time = time.time()
        r = m(self, *args, **kwargs)
        self._method_times.append((m.__name__, time.time() - start_time))
        return r
    return m_timed
