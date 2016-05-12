# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from copy import deepcopy

import numpy as np

from astropy import log
from astropy.units import Unit, Quantity
from astropy.utils.metadata import MetaData

from .nddata_base import NDDataBase
from .nduncertainty import NDUncertainty, UnknownUncertainty


__all__ = ['NDData']


class NDData(NDDataBase):
    """
    A container for `numpy.ndarray`-based datasets, using the
    `~nddata.nddata.NDDataBase` interface.

    The key distinction from raw `numpy.ndarray` is the presence of
    additional metadata such as uncertainty, mask, unit, a coordinate system
    and/or a dictionary containg further meta information. This class *only*
    provides a container for *storing* such datasets. For further functionality
    take a look at the ``See also`` section.

    Parameters
    -----------
    data : `numpy.ndarray`-like or `NDData`-like
        The dataset.

    uncertainty : any type, optional
        Uncertainty in the dataset.
        Should have an attribute ``uncertainty_type`` that defines what kind of
        uncertainty is stored, for example ``"std"`` for standard deviation or
        ``"var"`` for variance. A metaclass defining such an interface is
        `NDUncertainty` - but isn't mandatory. If the uncertainty has no such
        attribute the uncertainty is stored as `UnknownUncertainty`.
        Defaults to ``None``.

    mask : any type, optional
        Mask for the dataset. Masks should follow the ``numpy`` convention that
        **valid** data points are marked by ``False`` and **invalid** ones with
        ``True``.
        Defaults to ``None``.

    wcs : any type, optional
        World coordinate system (WCS) for the dataset.
        Default is ``None``.

    meta : `dict`-like object, optional
        Additional meta informations about the dataset. If no meta is provided
        an empty `collections.OrderedDict` is created.
        Default is ``None``.

    unit : `~astropy.units.Unit`-like, optional
        Unit for the dataset. Strings that can be converted to a
        `~astropy.units.Unit` are allowed.
        Default is ``None``.

    copy : `bool`, optional
        Save the attributes as copy? ``True`` copies every attribute before
        saving it while ``False`` tries to save every parameter as reference.
        Note however that it is not always possible to save the input as
        reference.
        Default is ``False``.

        .. versionadded:: 1.2

    Raises
    ------
    TypeError
        In case ``data`` or ``meta`` don't meet the restrictions.

    Attributes
    ----------
    meta : `dict`-like
        Additional meta informations about the dataset.

    Notes
    -----
    Each attribute can be accessed through the homonymous instance attribute:
    ``data`` in a `NDData` object can be accessed through the `data`
    attribute::

        >>> from nddata.nddata import NDData
        >>> nd = NDData([1,2,3])
        >>> nd.data
        array([1, 2, 3])

    Given a conflicting implicit and an explicit parameter during
    initialization, for example the ``data`` is a `~astropy.units.Quantity` and
    the unit parameter is not None. Then the implicit parameter is replaced
    (without conversion) by the explicit one and a warning is issued::

        >>> import numpy as np
        >>> import astropy.units as u
        >>> q = np.array([1,2,3,4]) * u.m
        >>> nd2 = NDData(q, unit=u.cm)
        INFO: overwriting Quantity's current unit with specified unit. \
[nddata.nddata.nddata]
        >>> nd2.data
        array([ 1.,  2.,  3.,  4.])
        >>> nd2.unit
        Unit("cm")

    See also
    --------
    NDDataRef
    NDDataArray
    """

    def __init__(self, data, uncertainty=None, mask=None, wcs=None,
                 meta=None, unit=None, copy=False):

        # Rather pointless since the NDDataBase does not implement any setting
        # but before the NDDataBase did call the uncertainty
        # setter. But if anyone wants to alter this behaviour again the call
        # to the superclass NDDataBase should be in here.
        super(NDData, self).__init__()

        # The class name of the data parameter customize the info messages.
        name = data.__class__.__name__

        # Setup some temporary variables to hold implicitly passed arguments
        # so we can check for conflicts after collecting them.
        unit2 = None
        meta2 = None
        mask2 = None
        uncertainty2 = None
        wcs2 = None

        # Check if data is any type from which to collect some implicitly
        # passed parameters.
        if isinstance(data, NDData):  # don't use self.__class__ (issue #4137)
            # Another NDData object get all attributes
            unit2 = data.unit
            meta2 = data.meta
            mask2 = data.mask
            uncertainty2 = data.uncertainty
            wcs2 = data.wcs
            data = data.data
        elif hasattr(data, '__astropy_nddata__'):
            # Something that provides an interface to convert to NDData
            # collect the dictionary returned by it but assume not every
            # argument is provided, therefore use get with None default
            kwargs = data.__astropy_nddata__()
            # data is the only required argument here so let it fail if the
            # other class doesn't provide it.
            if 'data' not in kwargs:
                raise TypeError(
                    "missing data from interface class {0}".format(name))
            unit2 = kwargs.get('unit', None)
            meta2 = kwargs.get('meta', None)
            mask2 = kwargs.get('mask', None)
            uncertainty2 = kwargs.get('uncertainty', None)
            wcs2 = kwargs.get('wcs', None)
            data = kwargs['data']
        else:
            if hasattr(data, 'mask') and hasattr(data, 'data'):
                # Probably a masked array: Get mask and then data
                mask2 = data.mask
                data = data.data
            # It could be a masked quantity so no elif here.
            if isinstance(data, Quantity):
                # A quantity get the unit and the value.
                unit2 = data.unit
                data = data.value

        if any(not hasattr(data, attr)
                for attr in ('shape', '__getitem__', '__array__')):
            # Data doesn't look like a numpy array, try converting it to one.
            data2 = np.array(data, subok=True, copy=False)
        else:
            data2 = data

        # Another quick check to see if what we got looks like an array
        # rather than an object (since numpy will convert a
        # non-numerical/non-string inputs to an array of objects).
        if data2.dtype.kind not in 'buifc':
            raise TypeError(
                "could not convert {0} to numpy array.".format(name))

        # Check if explicit or implicit argument should be used and raise an
        # info if both are provided
        msg = "overwriting {0}'s current {1} with specified {1}."

        # Units are relativly cheap to compare so only raise the info message
        # if both are set and not equal. No need to compare the other arguments
        # though, especially since comparing numpy arrays could be expensive.
        if unit is not None and unit2 is not None and unit != unit2:
            log.info(msg.format(name, 'unit'))
        elif unit2 is not None:
            unit = unit2

        if mask is not None and mask2 is not None:
            log.info(msg.format(name, 'mask'))
        elif mask2 is not None:
            mask = mask2

        if meta and meta2:  # check if it's not empty here!
            log.info(msg.format(name, 'meta'))
        elif meta2:
            meta = meta2

        if wcs is not None and wcs2 is not None:
            log.info(msg.format(name, 'wcs'))
        elif wcs2 is not None:
            wcs = wcs2

        if uncertainty is not None and uncertainty2 is not None:
            log.info(msg.format(name, 'uncertainty'))
        elif uncertainty2 is not None:
            uncertainty = uncertainty2

        # Copy if necessary
        if copy:
            # only copy data if it wasn't converted before.
            if data is data2:
                data2 = deepcopy(data2)
            # always copy these mask, wcs and uncertainty
            mask = deepcopy(mask)
            wcs = deepcopy(wcs)
            uncertainty = deepcopy(uncertainty)
            # no need to copy meta because the meta descriptor will always copy
            # meta = deepcopy(meta)
            # and units don't need to be copied anyway.
            # unit = deepcopy(unit)

        # Store the attributes
        self._data = data2
        self.mask = mask
        self.wcs = wcs
        self.meta = meta
        self.unit = unit
        # Call the setter for uncertainty to further check the uncertainty
        self.uncertainty = uncertainty

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        prefix = self.__class__.__name__ + '('
        body = np.array2string(self.data, separator=', ', prefix=prefix)
        return ''.join([prefix, body, ')'])

    @property
    def data(self):
        """`numpy.ndarray`-like : The stored dataset.

        The data cannot be set directly but it probably can be modified
        in-place.
        """
        return self._data

    # Instead of a custom property use the MetaData descriptor. It will check
    # if the meta is dict-like.
    # TODO: reading the documentation from a descriptor using Sphinx isn't
    # trivial so this attribute is documented in the class docstring but
    # it would be better to define it here.
    meta = MetaData()

    @property
    def mask(self):
        """any type : Mask for the dataset, if any.

        Masks should follow the ``numpy`` convention that **valid** data points
        are marked by ``False`` and **invalid** ones with ``True``.
        """
        return self._mask

    @mask.setter
    def mask(self, value):
        self._mask = value

    @property
    def unit(self):
        """`~astropy.units.Unit` : Unit for the dataset, if any.

        .. warning::

          Setting or overwriting the unit manually will not check if the new
          unit is compatible or convertible to the old unit. Neither will this
          scale or otherwise affect the saved data or uncertainty. Appropriate
          conversion of these values must be done manually.
        """
        return self._unit

    @unit.setter
    def unit(self, value):
        # Simply replace the unit without converting data or uncertainty:
        if value is None:
            self._unit = None
        else:
            self._unit = Unit(value)

    @property
    def wcs(self):
        """any type : World coordinate system (WCS) for the dataset, if any.
        """
        return self._wcs

    @wcs.setter
    def wcs(self, value):
        self._wcs = value

    @property
    def uncertainty(self):
        """any type : Uncertainty in the dataset, if any.

        Should have an attribute ``uncertainty_type`` that defines what kind of
        uncertainty is stored, for example ``"std"`` for standard deviation or
        ``"var"`` for variance. A metaclass defining such an interface is
        `NDUncertainty` - but isn't mandatory. If the uncertainty has no such
        attribute the uncertainty is stored as `UnknownUncertainty`.
        """
        return self._uncertainty

    @uncertainty.setter
    def uncertainty(self, value):
        if value is not None:
            # There is one requirements on the uncertainty: That
            # it has an attribute 'uncertainty_type'.
            # If it does not match this requirement convert it to an unknown
            # uncertainty.
            if not hasattr(value, 'uncertainty_type'):
                log.info('uncertainty should have attribute uncertainty_type.')
                value = UnknownUncertainty(value, copy=False)

            # If it is a subclass of NDUncertainty we must set the
            # parent_nddata attribute. (#4152)
            if isinstance(value, NDUncertainty):
                # In case the uncertainty already has a parent create a new
                # instance because we need to assume that we don't want to
                # steal the uncertainty from another NDData object
                if value._parent_nddata is not None:
                    value = value.__class__(value, copy=False)
                # Then link it to this NDData instance (internally this needs
                # to be saved as weakref but that's done by NDUncertainty
                # setter).
                value.parent_nddata = self
        self._uncertainty = value
