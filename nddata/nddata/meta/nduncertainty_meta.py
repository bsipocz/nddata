# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from abc import ABCMeta, abstractproperty, abstractmethod
from ...utils.copyutils import do_copy
import weakref

import numpy as np

from astropy.extern import six

from astropy import log
import astropy.units as u

from ...utils import descriptors
from ...utils.sentinels import ParameterNotSpecified
from ..exceptions import MissingDataAssociationException

__all__ = ['NDUncertainty',
           'NDUncertaintyGaussian',
           'NDUncertaintyPropagatable']


@six.add_metaclass(ABCMeta)
class NDUncertainty(object):
    """Base metaclass (`abc.ABCMeta`) for uncertainty classes used with \
            `~.NDDataBase`.

    Parameters
    ----------
    data : any type, optional
        The array or value of the uncertainty. `numpy.ndarray`,
        `~astropy.units.Quantity` or `NDUncertainty`-like data is recommended.
        Default is ``None``.

    unit : `~astropy.units.Unit`, `str` or `None`, optional
        Unit for the uncertainty ``data``. Strings that can be converted to a
        `~astropy.units.Unit` are allowed.
        Default is ``None``.

    parent_nddata : `~.NDDataBase`-like or None, optional
        The parent dataset for this uncertainty.
        Defaults is ``None``.

    copy : `bool`, optional
        Indicates whether to save the ``data`` as a copy. ``True`` copies it
        before saving, while ``False`` tries to save every parameter as
        reference. Note however that it is not always possible to save the
        input as reference.
        Default is ``True``.

    Raises
    ------
    IncompatibleUncertaintiesException
        If given another `NDUncertainty`-like class and it isn't convertible.
    """

    def __init__(self, data=None, unit=ParameterNotSpecified,
                 parent_nddata=ParameterNotSpecified, copy=False):

        # Remember the class of the original data parameter. This will be
        # needed for Warnings or Exceptions later.
        name = data.__class__.__name__

        # Temporary variables to save implicitly passed parameters, appended a
        # 2 to clearly distinguish them to the explictly passed ones.
        unit2 = None
        parent_nddata2 = None

        # Catch the cases where the "data" has a special class to extract the
        # implicit parameters.

        if isinstance(data, NDUncertainty):
            # The data is another instance or subclass of NDUncertainty. Call
            # the conversion method so we can be sure it's the same class or
            # a convertable class.
            data = self.from_uncertainty(data)
            # Getting the parent_nddata is raises an Exception if no parent was
            # set so this must be done with try ... except
            try:
                parent_nddata2 = data.parent_nddata
            except MissingDataAssociationException:
                parent_nddata2 = None
            # Unit and data are straightforward.
            unit2 = data.unit
            data = data.data

        elif isinstance(data, u.Quantity):
            # The data is a Quantity, so we extract the unit and value
            unit2 = data.unit
            data = data.value

        # Resolve which parameter should be kept: implicit or explicit. See
        # NDData.__init__ for an extensive explanation why this is implemented
        # this way.

        msg = "overwriting {0}'s current {1} with specified {1}."

        if unit is ParameterNotSpecified:
            unit = unit2
        elif unit2 is not None and unit != unit2:
            log.info(msg.format(name, 'unit'))

        if parent_nddata is ParameterNotSpecified:
            parent_nddata = parent_nddata2
        elif (parent_nddata2 is not None and
                parent_nddata is not parent_nddata2):
            log.info(msg.format(name, 'parent_nddata'))

        self.data = data

        # only copy the data if it wasn't copied already during setting. See
        # this avoids unnecessary copy-steps. Hopefully noone does just a
        # shallow copy in the property...
        if copy and self.data is data:
            self.data = do_copy(data)
            # No need to copy unit because they are immutable
            # and copying parent_nddata would be bad since this would copy the
            # associated NDData instance!!!

        self.unit = unit
        self.parent_nddata = parent_nddata

    @classmethod
    def from_uncertainty(cls, uncertainty):
        """Converts an `~nddata.nddata.meta.NDUncertainty` instance to this \
                class.

        Parameters
        ----------
        uncertainty : `~nddata.nddata.meta.NDUncertainty`-like
            The uncertainty that should be converted.

        Returns
        -------
        converted_uncertainty : cls
            The converted uncertainty

        Raises
        ------
        IncompatibleUncertaintiesException
            In case the uncertainty cannot be converted to this class.

        Notes
        -----
        If the ``uncertainty`` has the same class then it is simply returned.
        Otherwise the `~.UncertaintyConverter` is used. If there are no
        registered functions to convert the uncertainties this will fail.
        """
        # If it's already the same class just return the uncertainty again.
        if uncertainty.__class__ is cls:
            return uncertainty

        from ..nduncertainty_converter import UncertaintyConverter

        # Get the appropriate function to convert between these classes. The
        # converter will raise an appropriate Exception if there is no
        # registered conversion function.
        cls2 = uncertainty.__class__
        func = UncertaintyConverter.get_converter_func(cls2, cls)

        # It might be a bit restrictive to assume that the converter does all
        # the determination and returns a dictionary containing the new values
        # - and it might in some cases be simpler to just use a function that
        # does the conversion. That would certainly work for
        # stddev <-> variance but as soon as other data gets important, like
        # relative uncertainties that need the parents data, a simple function
        # approach would fail. So to allow more freedom in what the converter
        # can do this was designed as is. I don't think special casing
        # converters would clean this up... but if this approach proves to
        # complicated one could easily extend this.
        return cls(copy=False, **func(uncertainty))

    # Copy and deepcopy magic and a public copy method
    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo=None):
        return self.copy()

    def copy(self):
        """Returns a deepcopy of the class.

        Returns
        -------
        copy : `~nddata.nddata.meta.NDUncertainty`-like
            A deepcopy of the current instance of the same class.

        Notes
        -----
        This will just call the ``__init__`` with this instance being the
        "data" and "copy=True".
        """
        # Since the __init__ is capable of doing a complete copy in case the
        # data is a NDUncertainty instance just call it.
        return self.__class__(self, copy=True)

    def is_identical(self, other, strict=True):
        """Compares if two uncertainties are identical.

        See also :meth:`nddata.nddata.NDDataBase.is_identical` for more
        information.
        """
        if self is other:
            return True

        if strict:
            if self.__class__ is not other.__class__:
                return False

        # Wrap everything in a try/except so AttributeErrors can be catched
        try:
            if self.data is None or other.data is None:
                if not (self.data is None and other.data is None):
                    return False
            else:
                if strict:
                    # The uncertainty has 2 values we would like to compare:
                    # data and unit. The data can be a numpy.ndarray so special
                    # case this:
                    if isinstance(self.data, np.ndarray):
                        if self.data.shape != other.data.shape:
                            return False
                        if np.any(self.data != other.data):
                            return False
                    else:
                        if self.data != other.data:
                            return False

                    if self.unit != other.unit:
                        return False
                else:
                    # Not strict compares them as quantities:

                    # First check if they have an effective unit and only then
                    # default to normal unit.
                    try:
                        unit1 = self.effective_unit
                    except (AttributeError, MissingDataAssociationException):
                        # Either it has no effective_unit (only gaussian
                        # uncertainties have one) or it has it but had no
                        # parent...
                        unit1 = self.unit

                    try:
                        unit2 = other.effective_unit
                    except (AttributeError, MissingDataAssociationException):
                        unit2 = other.unit

                    if unit1 is None:
                        data1 = self.data * u.dimensionless_unscaled
                    else:
                        data1 = self.data * unit1

                    if unit2 is None:
                        data2 = other.data * u.dimensionless_unscaled
                    else:
                        data2 = other.data * unit2

                    if np.any(data1 != data2):
                        return False

        except AttributeError:
            return False

        # We made it, it's identical :-)
        return True

    # Representation and casting to string
    def __repr__(self):
        prefix = self.__class__.__name__ + '('
        try:
            body = np.array2string(self.data, separator=', ', prefix=prefix)
        except AttributeError:
            # In case it wasn't possible to use array2string because some
            # attribute wasn't numpy-ndarray like
            body = str(self.data)
        return ''.join([prefix, body, ')'])

    def __str__(self):
        return self.__repr__()

    # NumPy like slicing
    def __getitem__(self, item):
        # Slice the array but keep the current unit and discard the parent.
        # Discarding the parent has two reasons: If no parent was set we would
        # get an exceptions. Also if the parent was sliced the parent nddata
        # is going through it's init and there the parent is set.
        # We don't need sliced uncertainties linking to unsliced data.
        return self.__class__(self.data[item], unit=self.unit, copy=False)

    # Descriptor-properties
    @descriptors.UncertaintyData
    def data(self):
        """(any type) Uncertainty value.
        """

    @descriptors.Unit
    def unit(self):
        """(`~astropy.units.Unit`) The unit of the uncertainty, if any.

        .. warning::

          Setting or overwriting the unit manually will not check if the new
          unit is compatible or convertible to the old unit. Neither will this
          scale or otherwise affect the saved uncertainty. Appropriate
          conversion of these values must be done manually.
        """

    # Real properties.
    @abstractproperty
    def uncertainty_type(self):
        """(`abc.abstractproperty`) Short description of the type of \
                uncertainty.
        """

    @property
    def parent_nddata(self):
        """(`~.NDDataBase`-like) reference to `~.NDDataBase` instance with \
                this uncertainty.

        In case the reference is not set uncertainty propagation will not be
        possible since propagation might need the uncertain data besides the
        uncertainty.
        """
        message = "uncertainty is not associated with an NDData object."
        if self._parent_nddata is None:
            # There is no parent so raise a custom exception. It is guaranteed
            # that there is such an private attribute because the init did set
            # it to None. If there is no private attribute someone must have
            # messed with private attributes. He will get the AttributeError.
            # Don't mess with private attributes :-)
            raise MissingDataAssociationException(message)

        # The NDData is saved as weak reference so we must call it
        # to get the object the reference points to.
        if isinstance(self._parent_nddata, weakref.ref):
            return self._parent_nddata()

        # If it wasn't a weakref someone messed with the private attribute.
        # Print a Warning since maybe someone used it in a pre-astropy 1.2
        # manner.
        # TODO: This could be removed ... but wait a bit... better to have this
        # warning than to fail it.
        log.info("parent_nddata should be a weakref to an NDData object.")
        return self._parent_nddata

    @parent_nddata.setter
    def parent_nddata(self, value):
        if value is not None and not isinstance(value, weakref.ref):
            # Save a weak reference on the uncertainty that points to this
            # instance of NDData. Direct references should NOT be used:
            # https://github.com/astropy/astropy/pull/4799#discussion_r61236832
            value = weakref.ref(value)
        self._parent_nddata = value


@six.add_metaclass(ABCMeta)
class NDUncertaintyPropagatable(NDUncertainty):
    """Metaclass (`abc.ABCMeta`) that extends `NDUncertainty` and indicates \
            that instances allow uncertainty propatation.
    """

    @abstractproperty
    def supports_correlated(self):
        """(`abc.abstractproperty`) Supports uncertainty propagation with \
                correlated uncertainties?

        Should be a `bool`.
        """

    @abstractmethod
    def propagate(self, *args, **kwargs):
        """(`abc.abstractmethod`) Propagate uncertainties.

        No restrictions on the arguments.
        """


@six.add_metaclass(ABCMeta)
class NDUncertaintyGaussian(NDUncertaintyPropagatable):
    """Metaclass (`abc.ABCMeta`) that extens `NDUncertaintyPropagatable` and \
            indicates the propagation assumes gaussian uncertainties.

    This metaclass ensures that the ``data`` is always a `numpy.ndarray`.
    """

    @descriptors.PlainArrayData
    def data(self):
        """(`numpy.ndarray`) Uncertainty value.

        Gaussian uncertainties need the value to be numeric so it is cast to
        a `numpy.ndarray`. It's not possible to assign subclasses as ``data``.
        """

    @abstractproperty
    def effective_unit(self):
        """(`abc.abstractproperty`) The effective unit of the instance.

        If the own ``unit`` is not set the (converted) unit of the parent
        should be returned. This ensures that propagation methods don't need
        to check for those cases by themselves.
        """

    @abstractmethod
    def propagate(self, *args, **kwargs):
        """(`abc.abstractmethod`) Propagate uncertainties based on first order \
                gaussian error propagation.
        """

    def offset(self, pad_width):
        return self.__class__(np.lib.pad(self.data, pad_width, mode='constant',
                                         constant_values=0),
                              unit=self.unit, copy=False)
