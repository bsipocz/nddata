# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from abc import ABCMeta, abstractproperty, abstractmethod
from copy import deepcopy
import weakref

import numpy as np

from astropy.extern import six

from astropy import log
from astropy.units import Quantity

from ...utils import descriptors
from ...utils.sentinels import ParameterNotSpecified

__all__ = ['MissingDataAssociationException',
           'IncompatibleUncertaintiesException',
           'UncertaintyConverter',
           'NDUncertainty',
           'NDUncertaintyGaussian',
           'NDUncertaintyPropagatable']


class IncompatibleUncertaintiesException(Exception):
    """This exception should be used to indicate cases in which uncertainties
    with two different classes can not be propagated.
    """


class MissingDataAssociationException(Exception):
    """This exception should be used to indicate that an uncertainty instance
    has not been associated with a parent `~.NDDataBase` object.
    """


class UncertaintyConverter(object):
    converter = {}

    @classmethod
    def register(cls, source, target, forward, backward):
        cls.converter[(source, target)] = forward
        cls.converter[(target, source)] = backward

    @classmethod
    def get_converter_func(cls, source, target):
        try:
            return cls.converter[(source, target)]
        except KeyError:
            msg = "cannot convert {0} to {1}".format(source.__name__,
                                                     target.__name__)
            raise IncompatibleUncertaintiesException(msg)


@six.add_metaclass(ABCMeta)
class NDUncertainty(object):
    """This is the metaclass for uncertainty classes used with `~.NDDataBase`.

    .. warning::
        NDUncertainty is an abstract class and should *never* be instantiated
        directly.

    Parameters
    ----------
    array : any type, optional
        The array or value (the parameter name is due to historical reasons) of
        the uncertainty. `numpy.ndarray`, `~astropy.units.Quantity` or
        `NDUncertainty` subclasses are recommended.

        If the `array` is `list`-like or `numpy.ndarray`-like it will be cast
        to a plain `numpy.ndarray`.
        Default is ``None``.

    unit : `~astropy.units.Unit` or str, optional
        Unit for the uncertainty ``array``. Strings that can be converted to a
        `~astropy.units.Unit` are allowed.
        Default is ``None``.

    copy : `bool`, optional
        Indicates whether to save the `array` as a copy. ``True`` copies it
        before saving, while ``False`` tries to save every parameter as
        reference. Note however that it is not always possible to save the
        input as reference.
        Default is ``True``.

    Raises
    ------
    IncompatibleUncertaintiesException
        If given another `NDUncertainty`-like class as ``array`` if their
        ``uncertainty_type`` is different.
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
            data = data.array

        elif isinstance(data, Quantity):
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

        if copy:
            data = deepcopy(data)
            # No need to copy unit because they are immutable
            # and copying parent_nddata would be bad since this would copy the
            # associated NDData instance!!!

        self.array = data
        self.unit = unit
        self.parent_nddata = parent_nddata

    @classmethod
    def from_uncertainty(cls, uncertainty):
        # If it's already the same class just return the uncertainty again.
        if uncertainty.__class__ is cls:
            return uncertainty

        # Get the appropriate function to convert between these classes. The
        # converter will raise an appropriate Exception if there is no
        # registered conversion function.
        cls2 = uncertainty.__class__
        func = UncertaintyConverter.get_converter_func(cls2, cls)

        # Temporary variables for creating the new instance.
        data = None
        unit = None
        parent_nddata = None

        if uncertainty.array is not None:
            # Apply the function on the array and save the return.
            data = func(uncertainty.array)
        if uncertainty.unit is not None:
            # Units probably cannot handle the function that was applied to the
            # array but quantities can, so convert it to a quantity and take
            # the unit of the result. Unfortunatly this can be very slow...
            unit = func(1 * uncertainty.unit).unit

        # To get the current parent we need a try except otherwise we would
        # let the exception raise in case the other had no parent.
        try:
            parent_nddata = uncertainty.parent_nddata
        except MissingDataAssociationException:
            pass

        # Call the init of the class.
        return cls(data, unit, parent_nddata, copy=False)

    @abstractproperty
    def uncertainty_type(self):
        """(`str`) Short description of the type of uncertainty.

        Defined as abstract property so subclasses *have* to override this.
        """

    @property
    def array(self):
        """(`numpy.ndarray`) Uncertainty value.
        """
        return self._array

    @array.setter
    def array(self, value):
        if isinstance(value, (list, np.ndarray)):
            value = np.array(value, subok=False, copy=False)
        self._array = value

    @descriptors.Unit
    def unit(self):
        """(`~astropy.units.Unit`) The unit of the uncertainty, if any.

        .. warning::

          Setting or overwriting the unit manually will not check if the new
          unit is compatible or convertible to the old unit. Neither will this
          scale or otherwise affect the saved uncertainty. Appropriate
          conversion of these values must be done manually.
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
        try:
            if self._parent_nddata is None:
                raise MissingDataAssociationException(message)
            else:
                # The NDData is saved as weak reference so we must call it
                # to get the object the reference points to.
                if isinstance(self._parent_nddata, weakref.ref):
                    return self._parent_nddata()
                else:
                    log.info("parent_nddata should be a weakref to an NDData "
                             "object.")
                    return self._parent_nddata
                return self._parent_nddata
        except AttributeError:
            raise MissingDataAssociationException(message)

    @parent_nddata.setter
    def parent_nddata(self, value):
        if value is not None and not isinstance(value, weakref.ref):
            # Save a weak reference on the uncertainty that points to this
            # instance of NDData. Direct references should NOT be used:
            # https://github.com/astropy/astropy/pull/4799#discussion_r61236832
            value = weakref.ref(value)
        self._parent_nddata = value

    def __repr__(self):
        prefix = self.__class__.__name__ + '('
        try:
            body = np.array2string(self.array, separator=', ', prefix=prefix)
        except AttributeError:
            # In case it wasn't possible to use array2string because some
            # attribute wasn't numpy-ndarray like
            body = str(self.array)
        return ''.join([prefix, body, ')'])

    def __getitem__(self, item):
        return self.__class__(self.array[item], unit=self.unit, copy=False)


@six.add_metaclass(ABCMeta)
class NDUncertaintyPropagatable(NDUncertainty):

    @property
    def supports_correlated(self):
        """(`bool`) Supports uncertainty propagation with correlated \
                 uncertainties?
        """
        return False

    @abstractmethod
    def propagate(self, *args, **kwargs):
        """(`abc.abstractmethod`) Propagate uncertainties.
        """


@six.add_metaclass(ABCMeta)
class NDUncertaintyGaussian(NDUncertaintyPropagatable):

    @property
    def effective_unit(self):
        """(`~astropy.units.Unit`) The effective unit of the instance. If the \
                `unit` is not set the converted unit of the parent is used.
        """
        if self._unit is None:
            if (self._parent_nddata is None or
                    self.parent_nddata.unit is None):
                return None
            else:
                return self.parent_nddata.unit
        return self._unit

    @abstractmethod
    def propagate(self, *args, **kwargs):
        """(`abc.abstractmethod`) Propagate uncertainties based on first order \
                gaussian error propagation.
        """
