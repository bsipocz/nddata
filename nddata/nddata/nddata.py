# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from copy import deepcopy

import numpy as np

from astropy import log
from astropy.units import Unit, Quantity
# TODO: Could be omitted if astropy/#4921 is merged
from ..utils import descriptors
# from astropy.utils.metadata import MetaData

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
        Additional meta information about the dataset. If no meta is provided
        an empty `collections.OrderedDict` is created.
        Default is ``None``.

    unit : `~astropy.units.Unit`-like, optional
        Unit for the dataset. Strings that can be converted to a
        `~astropy.units.Unit` are allowed.
        Default is ``None``.

    copy : `bool`, optional
        Indicates whether to save the arguments as copy. ``True`` copies
        every attribute before saving it while ``False`` tries to save every
        parameter as reference.
        Default is ``False``.

    Raises
    ------
    TypeError
        In case ``data`` or ``meta`` don't meet the restrictions.

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
    the unit parameter is not ``None``, then the implicit parameter is replaced
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
            unit2 = kwargs.get('unit', None)
            meta2 = kwargs.get('meta', None)
            mask2 = kwargs.get('mask', None)
            uncertainty2 = kwargs.get('uncertainty', None)
            wcs2 = kwargs.get('wcs', None)
            data = kwargs.get('data', None)
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

        # At this point we know what the data is
        self.data = data
        if copy and self.data is data:
            self.data = deepcopy(data)

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

        # Copy if necessary (data is already copied)
        if copy:
            # always copy these mask, wcs and uncertainty
            mask = deepcopy(mask)
            wcs = deepcopy(wcs)
            uncertainty = deepcopy(uncertainty)
            # no need to copy meta because the meta descriptor will always copy
            # meta = deepcopy(meta)
            # and units don't need to be copied anyway.
            # unit = deepcopy(unit)

        # Store the attributes
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

    @descriptors.Data
    def data(self):
        """`numpy.ndarray`-like : The stored dataset.

        Only numerical arrays can be saved or ``None``.
        """

    # Instead of a custom property use the MetaData descriptor also used for
    # Tables. It will check if the meta is dict-like or raise an exception.
    @descriptors.Meta
    def meta(self):
        """`dict`-like : Additional meta information about the dataset.
        """

    @descriptors.Mask
    def mask(self):
        """any type : Mask for the dataset, if any.

        Masks should follow the ``numpy`` convention that **valid** data points
        are marked by ``False`` and **invalid** ones with ``True``.
        """

    @descriptors.Unit
    def unit(self):
        """`~astropy.units.Unit` : Unit for the dataset, if any.

        .. warning::

          Setting or overwriting the unit manually will not check if the new
          unit is compatible or convertible to the old unit. Neither will this
          scale or otherwise affect the saved data or uncertainty. Appropriate
          conversion of these values must be done manually.
        """

    @descriptors.WCS
    def wcs(self):
        """any type : World coordinate system (WCS) for the dataset, if any.
        """

    @descriptors.Uncertainty
    def uncertainty(self):
        """any type : Uncertainty in the dataset, if any.

        Should have an attribute ``uncertainty_type`` that defines what kind of
        uncertainty is stored, for example ``"std"`` for standard deviation or
        ``"var"`` for variance. A metaclass defining such an interface is
        `NDUncertainty` - but isn't mandatory. If the uncertainty has no such
        attribute the uncertainty is stored as `UnknownUncertainty`.
        """
