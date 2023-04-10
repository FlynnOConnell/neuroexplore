#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#wrappers.py.py
Module (utils): Wrappers to decorate for functionality.
"""

import functools
import time


def log_time(func):
    """Logs the time it took for func to execute"""
    def wrapper(*args, **kwargs):
        start = time.time_ns() / (10 ** 9)
        val = func(*args, **kwargs)
        end = time.time_ns() / (10 ** 9)
        duration = end - start
        print(f"{func.__name__} execution time: {duration}s")
        return val
    return wrapper


def typecheck(*args1, isclassmethod=False):
    """
    This typecheck function decorator to declare the type of each function or method argument.

    Example:
        @typecheck(types.StringType, Decimal)
        def my_function(s, d):
            pass

    Types: https://stackless.readthedocs.io/en/2.7-slp/library/types.html
    Cannot be used to typecheck n method argument that is of the same type as the method's
    class.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args2, **keywords):
            args = args2[1:] if isclassmethod else args2
            for (arg2, arg1) in zip(args, args1):
                if not isinstance(arg2, arg1):
                    raise TypeError(f"Expected type: {arg1}, actual type: {type(arg2)}")
            return func(*args2, **keywords)

        return wrapper

    return decorator


class Singleton:
    """
    A non-thread-safe helper class to ease implementing singletons.
    This should be used as a decorator -- not a metaclass -- to the
    class that should be a singleton.
    The decorated class can define one `__init__` function that
    takes only the `self` argument. Also, the decorated class cannot be
    inherited from. Other than that, there are no restrictions that apply
    to the decorated class.
    To get the singleton instance, use the `Instance` method. Trying
    to use `__call__` will result in a `TypeError` being raised.
    Ref: https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python
    """

    def __init__(self, decorated):
        self._decorated = decorated

    def Instance(self):
        """
        Returns the singleton instance. Upon its first call, it creates a
        new instance of the decorated class and calls its `__init__` method.
        On all subsequent calls, the already created instance is returned.
        """
        try:
            return self._instance
        except AttributeError:
            self._instance = self._decorated()
            return self._instance

    def __call__(self):
        raise TypeError("Singletons must be accessed through `Instance()`.")

    def __instancecheck__(self, inst):
        return isinstance(inst, self._decorated)