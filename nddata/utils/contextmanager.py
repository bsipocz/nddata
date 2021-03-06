# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from copy import deepcopy


__all__ = ['DictionaryContext']


class DictionaryContext(object):
    """A dictionary container that can be used as context manager.

    The context manager allows to modify the dictionary values and after
    exiting it resets them to the original state.

    Parameters
    ----------
    kwargs :
        Initial values for the contained dictionary.

    Attributes
    ----------
    dct : dict
        The `dict` containing the key-value pairs

    See also
    --------
    nddata.nddata.ContextArithmeticDefaults

    Examples
    --------
    By itself it is just a dictionary that can be modified globally but also
    set and reset in a local context. For example if you want to create a
    custom set of defaults and reset them afterwards::

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
        self.dct_copy = self.dct.copy()
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
