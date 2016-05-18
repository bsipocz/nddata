.. _nddata_interface:

Interface for NDData
====================

Classes that don't inherit from `~nddata.nddata.NDData` but want to be
convertible to one can implement a method ``__astropy_nddata__`` to provide
necessary keywords for creating a NDData instance or subclass.

Implementing the interface
--------------------------

The method ``__astropy_nddata__`` should take no necessary arguments and
must return a `dict`-like object including the necessary arguments to create
an NDData instance.

Complete interface
^^^^^^^^^^^^^^^^^^

A complete interface would be provided if all attributes were provided. This
would probably not be the case for real classes but provides a good example::

    >>> from nddata.nddata import NDData, NDDataRef

    >>> class NDDataInterface(object):
    ...
    ...     def __init__(self, data, uncertainty=None, mask=None, wcs=None,
    ...                  meta=None, unit=None, flags=None):
    ...         self.data = data
    ...         self.uncertainty = uncertainty
    ...         self.mask = mask
    ...         self.wcs = wcs
    ...         self.unit = unit
    ...         self.meta = meta
    ...         self.flags = flags
    ...
    ...     def __astropy_nddata__(self):
    ...         return {'data': self.data, 'uncertainty': self.uncertainty,
    ...                 'mask': self.mask, 'unit': self.unit, 'wcs': self.wcs,
    ...                 'meta': self.meta, 'flags': self.flags}
    ...
    >>> nddlike = NDDataInterface([1], 2, 3, 4, {1: 1}, 'm', True)
    >>> ndd = NDData(nddlike)
    INFO: uncertainty should have attribute uncertainty_type. [nddata.utils.descriptors]

    >>> ndd
    NDData([1])
    >>> ndd.mask
    3

    >>> ndd = NDDataRef(nddlike)
    INFO: uncertainty should have attribute uncertainty_type. [nddata.utils.descriptors]
    >>> ndd
    NDDataRef([1])

But given an explicit argument this will be used::

    >>> nddlike = NDDataInterface([1], mask=2)
    >>> ndd = NDDataRef(nddlike, mask=False)
    INFO: overwriting NDDataInterface's current mask with specified mask. [nddata.nddata.nddata]

    >>> ndd.mask
    False

.. note::
    This would also allow using it with ``NDData(**nddlike.__astropy_nddata__())``.

Partial interface
^^^^^^^^^^^^^^^^^

A partial interface would be more common. A partial interface only gives some
of the attributes (these could even be some that can only be used by
subclasses) and only the relevant ones are used::

    >>> class NDDataPartialInterface(NDDataInterface):
    ...     def __astropy_nddata__(self):
    ...         return {'data': self.data, 'unit': self.unit, 'meta': self.meta,
    ...                 'unneccesary': 'do not use me'}

    >>> nddlike = NDDataPartialInterface([1], 2, 3, 4, {1: 1}, 'm')
    >>> ndd = NDDataRef(nddlike)
    >>> ndd
    NDDataRef([1])

    >>> ndd.unit
    Unit("m")

    >>> ndd.uncertainty is None
    True

Broken interfaces
^^^^^^^^^^^^^^^^^

It will also work if the return doesn't include a value for the ``data``::

    >>> class NDDataBrokenInterface(NDDataInterface):
    ...     def __astropy_nddata__(self):
    ...         return {'meta': self.meta, 'uncertainty': self.uncertainty,
    ...                 'mask': self.mask, 'unit': self.unit, 'wcs': self.wcs}

    >>> nddlike = NDDataBrokenInterface([1], 2, 3, 4, {1: 1}, 'm')
    >>> ndd = NDDataRef(nddlike)
    INFO: uncertainty should have attribute uncertainty_type. [nddata.utils.descriptors]
    >>> ndd.data is None
    True
