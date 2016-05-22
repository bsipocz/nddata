# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from copy import deepcopy


__all__ = ['DictionaryContext']


class DictionaryContext(object):
    """A dictionary container that can be used as context manager.

    The context manager allows to modify the dictionary values and after
    exiting it resets them to the original state.

    Arguments
    ---------
    kwargs :
        Initial values for the contained dictionary.

    Examples
    --------
    In case you have some values that can be saved as `dict`-like object you
    could alter them globally, but this needs you to keep track of when you
    changed it and if you want to reset them again. Using it as context manager
    unburdens you, but you still can use it to modify values globally::

        >>> from nddata.utils.contextmanager import DictionaryContext
        >>> with DictionaryContext(a=10, b=2) as d:
        ...     print(d['a'])
        ...     print(d['b'])
        10
        2
        >>> 'a' in DictionaryContext.dct
        False
        >>> 'b' in DictionaryContext.dct
        False

    Creating a subclass with custom initial values::

        >>> class Other(DictionaryContext):
        ...     dct = {'a': 1}
        >>> Other.dct['a']
        1
        >>> Other.dct['a'] = 10
        >>> Other.dct['a']
        10
        >>> with Other(a=5) as d:
        ...     print(d['a'])
        5
        >>> Other.dct['a']  # after ending the context it is reset again.
        10
    """
    dct = {}

    def __init__(self, **kwargs):
        # Copy the original and update the current dictionary with the values
        # passed in.
        self.dct_copy = deepcopy(self.dct)
        self.dct.update(kwargs)

    def __enter__(self):
        # return the dictionary so one can catch it if one wants don't want to
        # always update the class attribute.
        return self.dct

    def __exit__(self, type, value, traceback):
        # clear the dictionary (in case someone added a new value) and update
        # it with the original values again
        self.dct.clear()
        self.dct.update(self.dct_copy)
