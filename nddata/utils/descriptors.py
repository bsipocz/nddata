# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from collections import OrderedDict, Mapping
from copy import deepcopy

import numpy as np

from .numpy import is_numeric_array


__all__ = ['MetaData']


class MetaData(object):
    """A descriptor for classes that have a ``meta`` property.

    This can be set to any valid `~collections.Mapping`.

    Parameters
    ----------
    doc : `str`, optional
        Documentation for the attribute of the class.
        Default is ``""``.

    copy : `bool`, optional
        If ``True`` the the value is deepcopied before setting, otherwise it
        is saved as reference.
        Default is ``True``.
    """
    def __init__(self, doc="", copy=True):
        self.__doc__ = doc
        self.copy = copy

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        if not hasattr(instance, '_meta'):
            instance._meta = OrderedDict()
        return instance._meta

    def __set__(self, instance, value):
        if value is None:
            instance._meta = OrderedDict()
        else:
            if isinstance(value, Mapping):
                if self.copy:
                    instance._meta = deepcopy(value)
                else:
                    instance._meta = value
            else:
                raise TypeError("meta attribute must be dict-like")


class NumericalNumpyData(object):
    """A descriptor for classes that have a ``meta`` property.

    This can be set to any valid `~collections.Mapping`.

    Parameters
    ----------
    attr : `str`
        Name of the property.

    doc : `str`, optional
        Documentation for the attribute of the class.
        Default is ``""``.
    """
    def __init__(self, attr, doc=""):
        self.attr = '_' + attr
        self.__doc__ = doc

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        return getattr(instance, self.attr, None)

    def __set__(self, instance, value):
        # Allow None as valid value.
        if value is not None:
            # Save the original class name for the error message if it
            # cannot be converted to an allowed numpy.ndarray
            name = value.__class__.__name__
            # NumPy array like means has these 3 attributes
            if any(not hasattr(value, attr)
                    for attr in ('shape', '__getitem__', '__array__')):
                # It doesn't look like a NumPy array so convert it to one.
                # don't allow subclasses because masks, unit or else are
                # already saved elsewhere.
                value = np.asarray(value)
            # Final check if the array is numeric. This will internally use
            # np.asarray again. This shouldn't be a problem in most cases but
            # if anyone finds a valid type and creating or setting data is slow
            # check if this function is the bottleneck.
            if not is_numeric_array(value):
                raise TypeError("could not convert {0} to numeric numpy array."
                                "".format(name))
        setattr(instance, self.attr, value)