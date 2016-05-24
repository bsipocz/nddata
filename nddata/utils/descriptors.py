# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from collections import OrderedDict, Mapping
from copy import deepcopy

import numpy as np

from astropy import log
from astropy.extern.six import string_types
import astropy.units as u

from .numpyutils import is_numeric_array


__all__ = ['BaseDescriptor', 'AdvancedDescriptor',
           'ArrayData', 'Mask', 'ArrayMask', 'Meta', 'Unit', 'WCS',
           'Flags', 'Uncertainty', 'UncertaintyData']


class BaseDescriptor(object):
    """A basic descriptor for a `property`-like attribute with getter, setter \
            and deleter.

    This descriptor can be used as decorator or to directly set a class
    attribute. It will save the attributes with a leading underscore (private)
    and, if set, copy the value while using the setter.

    Parameters
    ----------
    attr : `str` or `types.MethodType`
        Name of the property or a method that **must** not contain a body but
        can be used to determine the name and docstring of the attribute.

    doc : `str`, optional
        Documentation for the attribute of the class. If the ``attr`` is a
        method with a documentation this parameter will be ignored.
        Default is ``""``.

    copy : `bool`, optional
        If ``True`` setting the attribute will use a copy of the value instead
        of a reference.
        Default is ``False``.

    Examples
    --------
    This descriptor can be used to specify a class attribute directly::

        >>> from nddata.utils.descriptors import BaseDescriptor
        >>> class Test1(object):
        ...     data = BaseDescriptor('data', doc="Some data.", copy=True)

        >>> print(Test1.data.__doc__)
        Some data.

    or as descriptor so that the name and documentation can be given like for
    `property` with the difference that the method shouldn't contain any body::

        >>> class Test2(object):
        ...     @BaseDescriptor
        ...     def data(self):
        ...         '''Some data.'''

        >>> print(Test2.data.__doc__)
        Some data.

    but then it's not possible to specify the copy parameter and the default
    will be used (``False``).

    The ``get`` will return the private attribute or if that wasn't set it
    will return ``None``::

        >>> t1 = Test1()
        >>> t2 = Test2()

        >>> t1.data                # It wasn't set yet so it returns None
        >>> hasattr(t1, '_data')   # make sure there is no private attribute
        False

        >>> t1.data = 1            # set the attribute
        >>> t1.data                # the getter returns the set value
        1
        >>> t1._data               # value is saved in the private attribute
        1

    but it's generally not recommended to access the private attribute
    directly. It is just shown here to illustrate how it internally works.

    We've already seen the ``set``, but here it's important how the ``copy``
    parameter was set::

        >>> data = [1, 2, 3]       # lists are mutable
        >>> t1.data = data         # t1 copies while set
        >>> t2.data = data         # t2 does not
        >>> data[0] = 5            # change the original list
        >>> t1.data                # t1.data hasn't changed
        [1, 2, 3]
        >>> t2.data                # while t2.data is changed
        [5, 2, 3]

    The ``delete`` will just delete the private attribute::

        >>> hasattr(t1, '_data')   # it has some private data
        True
        >>> del t1.data            # now delete it.
        >>> hasattr(t1, '_data')   # the private attribute was deleted
        False
        >>> t1.data                # and the getter will return None

    .. note::
        Make sure you really have some strong arguments before using this
        descriptor since it's not nearly as efficient as using a normal
        class attribute or a `property` and behaves differently.

    Not using it as decorator also allows to customize the name of the
    private attribute::

        >>> class Testclass(object):
        ...     data = BaseDescriptor('notdata')
        >>> t = Testclass()
        >>> t.data = 10            # set the data
        >>> hasattr(t, '_data')    # it is not saved as _data
        False
        >>> t._notdata             # but as _nodata.
        10
    """
    def __init__(self, attr, doc="", copy=False):
        # To allow it to be used as decorator one needs to check if the
        # first attribute is a string or not. A string means it was used
        # explicitly otherwise it was used as decorator.
        if not isinstance(attr, string_types):
            # Not a string so take the documentation (if avaiable) and name
            # from the method.
            if attr.__doc__:
                doc = attr.__doc__
            attr = attr.__name__

        self.__doc__ = doc      # Set the documentation of the instance.
        self.attr = '_' + attr  # Add leading underscore to the attribute name
        self.copy = copy        # Copy while setting the attribute?

    def __get__(self, instance, owner=None):
        # If no instance is given return the descriptor instance. Very
        # important if one wants the attribute to be documented by sphinx!
        if instance is None:
            return self
        return getattr(instance, self.attr, None)

    def __set__(self, instance, value):
        if self.copy:
            # Copy it if necessary with deepcopy
            value = deepcopy(value)
        setattr(instance, self.attr, value)

    def __delete__(self, instance):
        delattr(instance, self.attr)


class AdvancedDescriptor(BaseDescriptor):
    """Like `BaseDescriptor` but allows to specify a default for the getter \
            and conditions for the setter.

    Parameters
    ----------
    args, kwargs :
        See `BaseDescriptor`.

    Notes
    -----
    Using this descriptor directly is discouraged. It's less efficient than
    `BaseDescriptor` and does the same. Only if you subclass it and override
    :meth:`create_default` or :meth:`process_value` it can offer an easily
    extendable attribute. But make sure you read the comments on the
    methods carefully when extending it.

    .. note::
        The methods `create_default` and `process_value` are public but
        normally they shouldn't be called directly. These should only serve
        documentation purposes to explain when and how the descriptor
        internally works.

    Examples
    --------
    Apart from small differences this descriptor works exactly like
    `BaseDescriptor`. The difference only shows what happens if you use the
    getter when the private attribute doesn't exist::

        >>> from nddata.utils.descriptors import (BaseDescriptor,
        ...                                       AdvancedDescriptor)
        >>> class Test(object):
        ...     data1 = BaseDescriptor('data1', copy=False)
        ...     data2 = AdvancedDescriptor('data2', copy=False)

        >>> t = Test()
        >>> t.data1                # access the getter.
        >>> t.data2
        >>> hasattr(t, '_data1')   # data1 has no private attribute.
        False
        >>> hasattr(t, '_data2')   # but data2 has.
        True
        >>> t._data2               # the default value (None) was saved.

    Also deleting the attribute again will not delete it but only reset it to
    the default value::

        >>> t.data1 = 1
        >>> t.data2 = 1
        >>> del t.data1
        >>> del t.data2
        >>> hasattr(t, '_data1')   # private data1 was deleted.
        False
        >>> hasattr(t, '_data2')   # but private data2 was not.
        True
        >>> t._data2               # the default value (None) was saved again.

    """
    def create_default(self):
        """Create a default value for the property and return it.

        It is called during:

        - ``__get__``: if there is no private attribute.

        - ``__set__``: if the value is ``None``.

        - ``__delete__``: always.

        - ``__get__``: if the private attribute is ``None`` but which should
          be nearly impossible except one sets the private attribute itself
          which is discouraged to ensure correct behaviour.

        The default value is set as private attribute after this function
        is called to ensure correct behaviour in case the default is a
        mutable object.

        Examples
        --------
        Some examples for subclassing::

            >>> from nddata.utils.descriptors import AdvancedDescriptor
            >>> class DefaultZero(AdvancedDescriptor):
            ...     def create_default(self):
            ...         return 0

            >>> class Test(object):
            ...     data = DefaultZero('data')

        The default value is created when the attribute wasn't set before::

            >>> t = Test()
            >>> t.data
            0

        And also if it is set to None::

            >>> t.data = 1
            >>> t.data
            1
            >>> t.data = None
            >>> t.data
            0

        And when the attribute is deleted::

            >>> t.data = 1
            >>> del t.data
            >>> t.data
            0
        """

    def _set_get_default(self, instance):
        """Method that creates the default, sets the private attribute and
        returns it.

        Should not be overridden in subclasses it's just here because to
        reduce code repetition.
        """
        # In case we are dealing with mutable types we need to return the
        # instance we saved!
        default = self.create_default()        # Create a default value
        setattr(instance, self.attr, default)  # save it as private attribute
        return default                         # and return it

    def process_value(self, instance, value):
        """Take the value do appropriate conversions or checks and return it.

        Parameters
        ----------
        instance, value : any type
            The two parameters given to ``__set__``.

        Returns
        -------
        verified_value : any type
            The checked or converted value that will be set.

        Notes
        -----
        If the setter is called ``a.x = v`` then ``a`` is the ``instance`` and
        ``v`` is the ``value``.

        Examples
        --------
        An example for subclassing it, for example we want the value to be
        converted to a numpy-array (see also `ArrayData` which essentially does
        just that)::

            >>> from nddata.utils.descriptors import AdvancedDescriptor
            >>> import numpy as np

            >>> class ArrayDescriptor(AdvancedDescriptor):
            ...     def process_value(self, instance, value):
            ...         return np.asarray(value)

            >>> class Test(object):
            ...     data1 = ArrayDescriptor('data', copy=False)
            ...     data2 = ArrayDescriptor('data', copy=True)

        While setting the value will be converted to a `numpy.ndarray`::

            >>> t = Test()

            >>> t.data1 = 1
            >>> t.data1
            array(1)

            >>> t.data1 = [1,2,3]
            >>> t.data1
            array([1, 2, 3])

            >>> t.data1 = np.arange(5)
            >>> t.data1
            array([0, 1, 2, 3, 4])

        But this method is not called if the value is ``None``::

            >>> t.data1 = None
            >>> t.data1

        One special property of this descriptor is that if it should copy
        during setting (like the ``data2`` class property above) it checks
        if the value is the same after ``process_value`` by checking if
        ``value_before is value_after`` and only copies it if the condition is
        ``True``.

        In this example the value isn't the same after ``process_value`` so it
        will not be copied again afterwards::

            >>> t.data2 = [1]
            >>> t.data2
            array([1])

        but it will be copied afterwards if it is the same::

            >>> t.data2 = np.array([1])
            >>> t.data2
            array([1])

        So make certain that if you alter the value in a way that it isn't the
        same afterwards that you really copied it in ``process_value``!

        Another common case is to check if the value respects some
        requirements. For example if it's only allowed to set numerical
        values::

            >>> from nddata.utils.descriptors import AdvancedDescriptor
            >>> from numbers import Number

            >>> class NumberDescriptor(AdvancedDescriptor):
            ...     def process_value(self, instance, value):
            ...         if not isinstance(value, Number):
            ...             raise TypeError()
            ...         return value  # otherwise return it

            >>> class Test(object):
            ...     data = NumberDescriptor('data')

            >>> t = Test()

            >>> t.data = 1
            >>> t.data
            1

            >>> t.data = 1.0
            >>> t.data
            1.0

            >>> t.data = 1+2j
            >>> t.data
            (1+2j)

            >>> try:
            ...     t.data = 'a'
            ... except TypeError:
            ...     print('failed to set attribute.')
            failed to set attribute.

        """
        return value

    def __get__(self, instance, owner=None):
        # Fetch the result, this makes also sure self is returned in case
        # it is called on the instance!
        result = super(AdvancedDescriptor, self).__get__(instance, owner)
        # the super had a default of None, so assume None means we should
        # create a default value.
        if result is None:
            result = self._set_get_default(instance)
        return result

    def __set__(self, instance, value):
        if value is None:
            # No need for super because default should make sure it doesn't
            # need to be copied!
            self._set_get_default(instance)
        else:
            # Check if the conditions are met. It's expected that this raises
            # an Exception if they are not met!
            v_value = self.process_value(instance, value)

            if self.copy and value is not v_value:
                # The value was altered during the conditions check so
                # disable copy for the time of the super call and set it again
                # afterwards.
                # TODO: Python isn't multithreaded normally but this could
                # be a problem if run in different threads. At least test it!
                self.copy = False
                super(AdvancedDescriptor, self).__set__(instance, v_value)
                self.copy = True
                # For debugging purposes
                # from astropy import log
                # log.debug('temporarly disabled copy because data was copied '
                #           'during process_value.')
            else:
                super(AdvancedDescriptor, self).__set__(instance, v_value)

    def __delete__(self, instance):
        # no need to super because we don't want to delete the attribute just
        # reset it.
        self._set_get_default(instance)


class WCS(BaseDescriptor):
    """A `BaseDescriptor` without any alterations.
    """
    pass


class Mask(BaseDescriptor):
    """A `BaseDescriptor` without any alterations.
    """
    pass


class Flags(BaseDescriptor):
    """A `BaseDescriptor` without any alterations.
    """
    pass


class UncertaintyData(BaseDescriptor):
    """A `BaseDescriptor` without any alterations.
    """
    pass


class Meta(AdvancedDescriptor):
    """An `AdvancedDescriptor` which defaults to `~collections.OrderedDict` \
            and checks if the value is a `~collections.Mapping`.

    Parameters
    ----------
    args, kwargs :
        see :class:`AdvancedDescriptor`.
    """
    def create_default(self):
        """Returns an empty `~collections.OrderedDict`.
        """
        return OrderedDict()

    def process_value(self, instance, value):
        """Checks if the value is a `~collections.Mapping`.

        Parameters
        ----------
        args, kwargs :
            see :meth:`AdvancedDescriptor.process_value`.

        Raises
        ------
        TypeError
            If the value is not a subclass of `~collections.Mapping`.

        Returns
        -------
        value : subclass of `~collections.Mapping`.
            The value that is being set as private attribute.
        """
        if not isinstance(value, Mapping):
            raise TypeError("attribute '{0}' must be dict-like"
                            "".format(self.attr))
        return value


class ArrayData(AdvancedDescriptor):
    """An `AdvancedDescriptor` which checks if the value looks like \
            numerical `numpy.ndarray` or converts it to one.

    Parameters
    ----------
    args, kwargs :
        see :class:`AdvancedDescriptor`.
    """
    def create_default(self):
        """No default value, this returns ``None``.
        """

    def process_value(self, instance, value):
        """Checks if the value is a `numpy.ndarray` or similar enough.

        Parameters
        ----------
        args, kwargs :
            see :meth:`AdvancedDescriptor.process_value`.

        Raises
        ------
        TypeError
            If the value is not a `numpy.ndarray` or a subclass.

        Returns
        -------
        value : `numpy.ndarray`-like
            The value that is being set as private attribute.

        See also
        --------
        nddata.utils.numpyutils.is_numeric_array

        Notes
        -----
        The criteria are that the value has the attributes ``shape``,
        ``__getitem__`` and ``__array__`` and the check if it's numerical
        requires that it has a valid ``dtype.kind``. Otherwise it is attempted
        to cast it with `numpy.asarray`.
        """

        # Save the original class name for the error message if it cannot be
        # converted to an allowed numpy.ndarray
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
        return value


class ArrayMask(AdvancedDescriptor):
    """An `AdvancedDescriptor` which checks if the value looks like \
            boolean `numpy.ndarray` or converts it to one.

    Parameters
    ----------
    args, kwargs :
        see :class:`AdvancedDescriptor`.

    Examples
    --------

    Creating a class using this descriptor is mostly identical to assigning a
    `property`, even though the setter and deleter are created automatically
    and cannot be overriden with the property syntax ``@mask.setter``. Note
    that like all `AdvancedDescriptor` the body of the attribute is ignored so
    leave it empty or insert a ``pass``::

        >>> from nddata.utils.descriptors import ArrayMask
        >>> class Test(object):
        ...     @ArrayMask
        ...     def mask(self):
        ...         '''Some documentation of the masks purpose.'''

    The documentation is kept::

        >>> Test.mask.__doc__
        'Some documentation of the masks purpose.'

    The setter will now always convert the input to a `numpy.ndarray` with
    dtype `bool`::

        >>> t = Test()
        >>> t.mask = True
        >>> t.mask
        array(True, dtype=bool)

    Notice that every Python object can be evaluated as boolean, but see for
    yourself::

        >>> t.mask = [True, False, 1, 'a']
        >>> t.mask
        array([ True, False,  True,  True], dtype=bool)

    .. note::
        One use would be to override the ``mask`` of `~nddata.nddata.NDData`
        or `~nddata.nddata.NDDataBase` if you want a more
        `numpy.ma.MaskedArray`-like behaviour and don't want to convert the
        mask yourself. Just import ``NDData`` (or ``NDDataBase``) and set the
        descriptor:
        ``NDData.mask = ArrayMask('mask', 'docstring', copy=False)``
        But be aware that this will change will affect all your ``NDData``
        instances in the current session!
    """
    def create_default(self):
        """No default value, this returns ``None``.
        """

    def process_value(self, instance, value):
        """Checks if the value is a `numpy.ndarray` of boolean type or casts \
                it to one.

        Parameters
        ----------
        args, kwargs :
            see :meth:`AdvancedDescriptor.process_value`.

        Returns
        -------
        value : `numpy.ndarray`-like
            The value that is being set as private attribute.
        """
        # Very simple, if it's already a numpy.ndarray with dtype bool return
        # it.
        if isinstance(value, np.ndarray) and value.dtype == bool:
            return value
        # If it wasn't convert it to one explicitly.
        return np.array(value, dtype=bool, copy=False, subok=False)


class Unit(AdvancedDescriptor):
    """An `AdvancedDescriptor` which converts the value to an \
            `~astropy.units.Unit`.

    Parameters
    ----------
    args, kwargs :
        see :class:`AdvancedDescriptor`.
    """
    def create_default(self):
        """No default value, this returns ``None``.
        """

    def process_value(self, instance, value):
        """Converts the value to a `~astropy.units.Unit`.

        Parameters
        ----------
        args, kwargs :
            see :meth:`AdvancedDescriptor.process_value`.

        Raises
        ------
        UnitConversionError
            If the value is not castable to `~astropy.units.Unit`.

        Returns
        -------
        value : `~astropy.units.Unit`
            The value that is being set as private attribute.
        """
        # Just convert it to a unit. This will raise an Exception if not
        # possible.
        return u.Unit(value)


class Uncertainty(AdvancedDescriptor):
    """An `AdvancedDescriptor` which ensures that \
            `~nddata.nddata.meta.NDUncertainty` is setup correctly as \
            uncertainty.

    Parameters
    ----------
    args, kwargs :
        see :class:`AdvancedDescriptor`.
    """
    def create_default(self):
        """No default value, this returns ``None``.
        """

    def process_value(self, instance, value):
        """Makes sure the uncertainty is setup correctly when set.

        Parameters
        ----------
        args, kwargs :
            see :meth:`AdvancedDescriptor.process_value`.

        Returns
        -------
        value : `~nddata.nddata.meta.NDUncertainty`-like
            The value that is being set as private attribute.

        Notes
        -----
        During ``__set__`` it checks if the value has an ``uncertainty_type``
        attribute. If it hasn't the value is wrapped as
        `~nddata.nddata.UnknownUncertainty`.

        Then if it's a subclass of `~nddata.nddata.meta.NDUncertainty` which
        already has a ``parent`` then it's wrapped as **reference** in another
        class (same class as before) so we have two uncertainties each linking
        to their own parent instead of stealing the ``parent``. Then the
        ``parent_nddata`` is set to the instance the setter was called on.
        """
        from ..nddata.meta import NDUncertainty
        # There is one requirements on the uncertainty: That
        # it has an attribute 'uncertainty_type'.
        # If it does not match this requirement convert it to an unknown
        # uncertainty.
        if not hasattr(value, 'uncertainty_type'):
            from ..nddata import UnknownUncertainty
            log.info('uncertainty should have attribute uncertainty_type.')
            # This wrapping would make the parents think that the value
            # was already copied so we must make sure it's copied here!
            value = UnknownUncertainty(value, copy=self.copy)

        # If it is a subclass of NDUncertainty we must set the
        # parent_nddata attribute. (#4152)
        if isinstance(value, NDUncertainty):
            # In case the uncertainty already has a parent create a new
            # instance because we need to assume that we don't want to
            # steal the uncertainty from another NDData object
            if value._parent_nddata is not None:
                # FIXME: Unfortunatly printing a log info pops up far too often
                # so there is no hint when a new uncertainty was created
                # because the old one already had a parent...
                # log.info('created another uncertainty because the '
                #          'uncertainty already had a parent.')
                # Copy it if necessary because the parent will think it was
                # already copied since it's another instance!
                value = value.__class__(value, copy=self.copy)
            # Then link it to this NDData instance (internally this needs
            # to be saved as weakref but that's done by NDUncertainty
            # setter).
            value.parent_nddata = instance
        return value
