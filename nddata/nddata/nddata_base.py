# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from copy import deepcopy

import numpy as np

from astropy import log
from astropy.units import Quantity

from .meta.nddata_meta import NDDataMeta
from ..utils import descriptors
from ..utils.sentinels import ParameterNotSpecified


__all__ = ['NDDataBase']


class NDDataBase(NDDataMeta):
    """
    A container for `numpy.ndarray`-based datasets, using the
    `~meta.NDDataMeta` interface.

    The key distinction from raw `numpy.ndarray` is the presence of
    additional metadata such as uncertainty, mask, unit, a coordinate system
    and/or a dictionary containg further meta information. This class *only*
    provides a container for *storing* such datasets. For further functionality
    take a look at the ``See also`` section.

    Parameters
    -----------
    data : `numpy.ndarray`-like or `NDDataBase`-like or None
        The dataset.

    uncertainty : any type, optional
        Uncertainty in the dataset.
        Should have an attribute ``uncertainty_type`` that defines what kind of
        uncertainty is stored, for example ``"std"`` for standard deviation or
        ``"var"`` for variance. A metaclass defining such an interface is
        `~meta.NDUncertainty` - but isn't mandatory. If the uncertainty has no
        such attribute the uncertainty is stored as `UnknownUncertainty`.
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

    flags : any type, optional
        Flags for the dataset.

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

        >>> from nddata.nddata import NDDataBase
        >>> nd = NDDataBase([1,2,3])
        >>> nd.data
        array([1, 2, 3])

    Given a conflicting implicit and an explicit parameter during
    initialization, for example the ``data`` is a `~astropy.units.Quantity` and
    the unit parameter is not ``None``, then the implicit parameter is replaced
    (without conversion) by the explicit one and a warning is issued::

        >>> import numpy as np
        >>> import astropy.units as u
        >>> q = np.array([1,2,3,4]) * u.m
        >>> nd2 = NDDataBase(q, unit=u.cm)
        INFO: overwriting Quantity's current unit with specified unit. \
[nddata.nddata.nddata_base]
        >>> nd2.data
        array([ 1.,  2.,  3.,  4.])
        >>> nd2.unit
        Unit("cm")
    """

    def __init__(self, data, uncertainty=ParameterNotSpecified,
                 mask=ParameterNotSpecified, wcs=ParameterNotSpecified,
                 meta=ParameterNotSpecified, unit=ParameterNotSpecified,
                 flags=ParameterNotSpecified, copy=False):

        # In case the superclass implements some logic (currently it doesn't)
        # uncomment the following line.
        # super(NDData, self).__init__()

        # Remember the class of the original data parameter. This will be
        # needed for Warnings or Exceptions later.
        name = data.__class__.__name__

        # A set of temporary variables that could contain implicitly passed
        # attributes in case the "data" was some class which provided
        # additional attributes. The variable names have an appended 2 to
        # highlight the destinction between explicit and implicit parameters.
        unit2 = None
        meta2 = None
        mask2 = None
        uncertainty2 = None
        wcs2 = None
        flags2 = None

        # Now check if the "data" is something that provides implicitly given
        # parameters.
        if isinstance(data, NDDataBase):
            # Another NDData instance or subclass instance. Extract all
            # properties that the base NDData has.
            unit2 = data.unit
            meta2 = data.meta
            mask2 = data.mask
            uncertainty2 = data.uncertainty
            wcs2 = data.wcs
            flags2 = data.flags
            data = data.data

        elif hasattr(data, '__astropy_nddata__'):
            # Something that provides an interface to convert to NDData.
            # This is done after the "isinstance" check so that even if a
            # subclass implements the interface it isn't used. Maybe that's a
            # bad idea and it would be cleaner if these two checks are swapped.
            # It is expected that the interface returns a dictionary containing
            # valid implicit parameters. Only extract those that are needed to
            # setup a base class and use "get" in case the interface doesn't
            # provide all parameters.
            kwargs = data.__astropy_nddata__()
            unit2 = kwargs.get('unit', None)
            meta2 = kwargs.get('meta', None)
            mask2 = kwargs.get('mask', None)
            uncertainty2 = kwargs.get('uncertainty', None)
            wcs2 = kwargs.get('wcs', None)
            flags2 = kwargs.get('flags', None)
            data = kwargs.get('data', None)

        else:
            # It is neither a NDData instance nor does it implement an
            # interface. There are still two cases that might be of interest:
            if hasattr(data, 'mask') and hasattr(data, 'data'):
                # It has a "mask" and a "data" attribute. So it looks like a
                # numpy.ma.MaskedArray. So extract these two.
                mask2 = data.mask
                data = data.data

            # It is intentional that here is no "elif" because we might have
            # a masked Quantity here.
            if isinstance(data, Quantity):
                # It is an astropy Quantity, we can use the unit and the value.
                # Maybe it would be better to check for "value" and "unit"
                # attribute...
                unit2 = data.unit
                data = data.value

        # Now we have processed the implicit parameters and the data is
        # fixed. We want the data to be numpy array like or cast it to one, so
        # call the setter which takes care of this.
        self.data = data

        # The setter might copy and we have a copy argument. To avoid copying
        # the data twice we need some way of determining if the data was
        # copied. I assume that it was only copied if it isn't the same object
        # anymore. In case some data-type makes trouble I've left the debugging
        # code in here that prints a debug message in case the data was
        # altered. In practice this case is far too often to spam users with
        # this message:
        # data_unchanged = self.data is data
        # data_is_number = isinstance(data, (bool, int, float, complex))
        # if not data_unchanged and not copy and not data_is_number:
        #     print a debug message. This isn't interesting to anyone who
        #     isn't debugging.
        #     log.debug('the data was altered and probably copied to fulfill '
        #               'the restrictions of NDData.')

        # The setter may have already copied the data so check if the data has
        # changed and only copy the data again if "copy=True" was set and the
        # data wasn't changed.
        if copy and self.data is data:
            self.data = deepcopy(data)

        msg = "overwriting {0}'s current {1} with specified {1}."

        # Except for the data each other parameter might be given explicitly
        # and/or implicit. The approach here is easy only in case the explicit
        # parameter was not specified at all (ParameterNotSpecified sentinel)
        # the implicit parameter is used.

        # But to print an info message in case of potential conflics I also
        # check the case when there is an implicit parameter. Using elif
        # ensures that this message only comes if there was some explicit
        # parameter set (even if it was only None).

        # It's not a problem here but another approach might not replace the
        # ParameterNotSpecified value. Then it would bubble up to the user
        # who shouldn't be bothered with it. Remember this in case you
        # change anything in the next lines.

        if unit is ParameterNotSpecified:
            unit = unit2
        elif unit2 is not None and unit != unit2:
            # This case differs somewhat from the following since I explicitly
            # compare if the explicit and implicit unit are different. This is
            # because comparing the unit is much cheaper than comparing
            # potential numpy-arrays and the value of the comparison is useable
            # in the boolean context of "and". Numpy arrays wouldn't.
            log.info(msg.format(name, 'unit'))

        if mask is ParameterNotSpecified:
            mask = mask2
        elif mask2 is not None:
            log.info(msg.format(name, 'mask'))

        if meta is ParameterNotSpecified:
            meta = meta2
        elif meta2 is not None:
            log.info(msg.format(name, 'meta'))

        if wcs is ParameterNotSpecified:
            wcs = wcs2
        elif wcs2 is not None:
            log.info(msg.format(name, 'wcs'))

        if uncertainty is ParameterNotSpecified:
            uncertainty = uncertainty2
        elif uncertainty2 is not None:
            log.info(msg.format(name, 'uncertainty'))

        if flags is ParameterNotSpecified:
            flags = flags2
        elif flags2 is not None:
            log.info(msg.format(name, 'flags'))

        # TODO: Except for the unit these steps are very similar. It might be
        # good to create a function that does this but given that the code is
        # straightforward to read this way - no need to rush. In case this
        # changes make sure using an external function doesn't prove to be
        # a potential bottleneck. The parameters might be expensive to throw
        # around...

        # At this point we know which parameters should be saved on in the
        # instance (we don't need the variables ending with 2 anymore).
        # The setter of the attributes always save these as reference (except
        # maybe the uncertainty internals) so we need to copy them before we
        # set them. This avoids using the setter twice.
        if copy:
            mask = deepcopy(mask)
            wcs = deepcopy(wcs)
            uncertainty = deepcopy(uncertainty)
            meta = deepcopy(meta)
            flags = deepcopy(flags)
            # one exception is the unit. Units seem to be immutable so we don't
            # bother copying it.
            # unit = deepcopy(unit)

        # Now call the respective setters. Order shouldn't matter (not like
        # astropy.nddata.NDData which needed as specific order).
        self.mask = mask
        self.wcs = wcs
        self.meta = meta
        self.unit = unit
        self.uncertainty = uncertainty
        self.flags = flags

    # Define how these classes are represented or cast to string. This will
    # only display the data, because we might be dealing with other attributes
    # that are big numpy-arrays and printing all set attributes might yield
    # a very long string.
    # These methods are based on numpy.array2string so if the "data" is NOT
    # a numpy-array (which shouldn't happen normally) these might fail. So
    # subclasses defining other data restrictions probably need to override
    # these methods.
    def __repr__(self):
        prefix = self.__class__.__name__ + '('
        body = np.array2string(self.data, separator=', ', prefix=prefix)
        return ''.join([prefix, body, ')'])

    def __str__(self):
        return self.__repr__()

    # Overwrite __copy__ and __deepcopy__ so that these commands setup the
    # returned class correctly. This is important especially because of the
    # uncertainty.parent_nddata which would otherwise link to the original
    # class. Internally they just use the also defined public copy method.
    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo=None):
        return self.copy()

    def copy(self):
        """Returns a deepcopy of the class.

        Returns
        -------
        copy : `NDDataBase`-like
            A deepcopy of the current instance of the same class.

        Notes
        -----
        This will just call the ``__init__`` with this instance being the
        "data" and "copy=True".
        """
        # Since the __init__ is capable of doing a complete copy in case the
        # data is a NDData instance just call it.
        return self.__class__(self, copy=True)

    # Define the attributes. The body of each of these attributes is empty
    # because the complete logic is inside the descriptors (used as decorators
    # here).
    @descriptors.Data
    def data(self):
        """(`numpy.ndarray`-like) The stored dataset.

        Only numerical arrays can be saved or ``None``.
        """

    @descriptors.Meta
    def meta(self):
        """(`dict`-like) Additional meta information about the dataset.
        """

    @descriptors.Mask
    def mask(self):
        """(any type) Mask for the dataset, if any.

        Masks should follow the ``numpy`` convention that **valid** data points
        are marked by ``False`` and **invalid** ones with ``True``.
        """

    @descriptors.Unit
    def unit(self):
        """(`~astropy.units.Unit`) Unit for the dataset, if any.

        .. warning::

          Setting or overwriting the unit manually will not check if the new
          unit is compatible or convertible to the old unit. Neither will this
          scale or otherwise affect the saved data or uncertainty. Appropriate
          conversion of these values must be done manually.
        """

    @descriptors.WCS
    def wcs(self):
        """(any type) World coordinate system (WCS) for the dataset, if any.
        """

    @descriptors.Flags
    def flags(self):
        """(any type) Flags for the dataset, if any.
        """

    @descriptors.Uncertainty
    def uncertainty(self):
        """(any type) Uncertainty in the dataset, if any.

        Should have an attribute ``uncertainty_type`` that defines what kind of
        uncertainty is stored, for example ``"std"`` for standard deviation or
        ``"var"`` for variance. A metaclass defining such an interface is
        `~meta.NDUncertainty` - but isn't mandatory. If the uncertainty has no
        such attribute the uncertainty is stored as `UnknownUncertainty`.
        """
