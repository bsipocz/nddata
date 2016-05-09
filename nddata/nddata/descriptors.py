# Licensed under a 3-clause BSD style license - see LICENSE.rst

from collections import OrderedDict, Mapping

from weakref import ref

from .exceptions import MissingParentNDDataException

__all__ = ['MetaDescriptor', 'ParentNDDataDescriptor']


class MetaDescriptor(object):
    """Descriptor for a `collections.Mapping`-like meta property that defaults
    to an empty `collections.OrderedDict`.

    Raises
    ------
    TypeError
        If the attribute is not a Mapping (setter).
    """
    def __get__(self, obj, objtype=None):
        return getattr(obj, '_meta', OrderedDict())

    def __set__(self, obj, value):
        if value is None:
            value = OrderedDict()
        elif not isinstance(value, Mapping):
            raise TypeError("meta attribute must be dict-like")
        obj._meta = value


class ParentNDDataDescriptor(object):
    """Descriptor that saves a `weakref.ref` reference parent_nddata property.

    Raises
    ------
    TypeError
        If the private attribute is not a weakref (getter).

    MissingParentNDDataException
        If the private attribute is None doesn't exist (getter).
    """
    def __get__(self, obj, objtype=None):
        result = getattr(obj, '_parent_nddata', None)
        if result is None:
            raise MissingParentNDDataException('Uncertainty has no associated '
                                               'NDData object.')
        elif not isinstance(result, ref):
            # The setter automatically saves a weakref so this Warning should
            # be only emitted if someone manually set the private attribute.
            raise TypeError('parent_nddata should be a weakref.ref object to '
                            'avoid circular references.')
        # Weakrefs must be called to yield the object they point to.
        return result()

    def __set__(self, instance, value):
        if value is not None:
            value = ref(value)
        setattr(instance, '_parent_nddata', value)
