.. _nddata_subclassing:

Subclassing
===========

`~nddata.nddata.NDDataBase`
---------------------------

This class serves as the base for subclasses that use a `numpy.ndarray` (or
something that presents a numpy-like interface) as the ``data`` attribute.

.. note::
  Each attribute is saved as attribute with one leading underscore. For example
  the ``data`` is saved as ``_data`` and the ``mask`` as ``_mask``, and so on.

Adding another property
^^^^^^^^^^^^^^^^^^^^^^^

    >>> from nddata.nddata import NDDataBase
    >>> from copy import deepcopy

    >>> class NDDataWithBlob(NDDataBase):
    ...     def __init__(self, *args, **kwargs):
    ...         # We allowed args and kwargs so find out where the data is.
    ...         data = args[0] if args else kwargs['data']
    ...
    ...         # There are three ways to get blob:
    ...         # 1.) explicitly given if they are given they should be used.
    ...         blob = kwargs.pop('blob', None)
    ...         if blob is None:
    ...             # 2.) another NDData - maybe with blob
    ...             if isinstance(data, NDDataBase):
    ...                 blob = getattr(data, 'blob', None)
    ...             # 3.) implements the NDData interface, maybe returns blob
    ...             elif hasattr(data, '__astropy_nddata__'):
    ...                 blob = data.__astropy_nddata__().get('blob', None)
    ...
    ...         # make sure the flags are copied if copy is set.
    ...         if 'copy' in kwargs and kwargs['copy']:
    ...             blob = deepcopy(blob)
    ...         # afterwards set it and call the parents init
    ...         self.blob = blob
    ...         super(NDDataWithBlob, self).__init__(*args, **kwargs)
    ...
    ...     @property
    ...     def blob(self):
    ...         return self._blob
    ...
    ...     @blob.setter
    ...     def blob(self, value):
    ...         self._blob = value

    >>> ndd = NDDataWithBlob([1,2,3])
    >>> ndd.blob is None
    True

    >>> ndd = NDDataWithBlob([1,2,3], blob=[0, 0.2, 0.3])
    >>> ndd.blob
    [0, 0.2, 0.3]

.. note::
  To simplify subclassing each setter (except for ``data``) is called during
  ``__init__`` so putting restrictions on any attribute can be done inside
  the setter and will also apply duing instance creation.

Customize the setter for a property
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    >>> import numpy as np

    >>> class NDDataMaskBoolNumpy(NDDataBase):
    ...
    ...     @property
    ...     def mask(self):
    ...         return self._mask
    ...
    ...     @mask.setter
    ...     def mask(self, value):
    ...         # Convert mask to boolean numpy array.
    ...         self._mask = np.array(value, dtype=np.bool_)

    >>> ndd = NDDataMaskBoolNumpy([1,2,3], mask=True)
    >>> ndd.mask
    array(True, dtype=bool)

Extend the setter for a property
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``unit``, ``meta`` and ``uncertainty`` implement some additional logic in their
setter so subclasses might define a call to the superclass and let the
super property set the attribute afterwards::

    >>> import numpy as np

    >>> class NDDataUncertaintyShapeChecker(NDDataBase):
    ...     @property
    ...     def uncertainty(self):
    ...         return self._uncertainty
    ...
    ...     @uncertainty.setter
    ...     def uncertainty(self, value):
    ...         value = np.asarray(value)
    ...         if value.shape != self.data.shape:
    ...             raise ValueError('uncertainty must have the same shape as the data.')
    ...         # Call the setter of the super class in case it might contain some
    ...         # important logic (only True for meta, unit and uncertainty)
    ...         super(NDDataUncertaintyShapeChecker, self.__class__).uncertainty.__set__(self, value)

    >>> ndd = NDDataUncertaintyShapeChecker([1,2,3], uncertainty=[2,3,4])
    INFO: uncertainty should have attribute uncertainty_type. [nddata.utils.descriptors]
    >>> ndd.uncertainty
    UnknownUncertainty([2, 3, 4])

`~nddata.nddata.NDData`
-----------------------

`~nddata.nddata.NDData` itself inherits from `~nddata.nddata.NDDataBase` so
any of the possibilities there also apply to NDData. But NDData also
inherits from the Mixins:

- `~nddata.nddata.mixins.NDSlicingMixin`
- `~nddata.nddata.mixins.NDArithmeticMixin`
- `~nddata.nddata.mixins.NDIOMixin`

which allow additional operations.

Slicing an existing property
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Suppose you have a class expecting a 2 dimensional ``data`` but the mask is
only 1D. This would lead to problems if one were to slice in two dimensions.

    >>> from nddata.nddata import NDData
    >>> import numpy as np

    >>> class NDDataMask1D(NDData):
    ...     def _slice_mask(self, item):
    ...         # Multidimensional slices are represented by tuples:
    ...         if isinstance(item, tuple):
    ...             # only use the first dimension of the slice
    ...             return self.mask[item[0]]
    ...         # Let the superclass deal with the other cases
    ...         return super(NDDataMask1D, self)._slice_mask(item)

    >>> ndd = NDDataMask1D(np.ones((3,3)), mask=np.ones(3, dtype=bool))
    >>> nddsliced = ndd[1:3,1:3]
    >>> nddsliced.mask
    array([ True,  True], dtype=bool)

.. note::
  The methods doing the slicing of the attributes are prefixed by a
  ``_slice_*`` where ``*`` can be ``mask``, ``uncertainty`` or ``wcs``. So
  simply overriding them is the easiest way to customize how the are sliced.

.. note::
  If slicing should affect the ``unit`` or ``meta`` see the next example.


Slicing an additional property
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Building on the added property ``blob`` we want them to be sliceable:

    >>> # The init and property is identical to the one earlier mentioned.
    >>> class NDDataWithBlob(NDData):
    ...     def __init__(self, *args, **kwargs):
    ...         # We allowed args and kwargs so find out where the data is.
    ...         data = args[0] if args else kwargs['data']
    ...
    ...         # There are three ways to get blob:
    ...         # 1.) explicitly given if they are given they should be used.
    ...         blob = kwargs.pop('blob', None)
    ...         if blob is None:
    ...             # 2.) another NDData - maybe with blob
    ...             if isinstance(data, NDDataBase):
    ...                 blob = getattr(data, 'blob', None)
    ...             # 3.) implements the NDData interface, maybe returns blob
    ...             elif hasattr(data, '__astropy_nddata__'):
    ...                 blob = data.__astropy_nddata__().get('blob', None)
    ...
    ...         # make sure the flags are copied if copy is set.
    ...         if 'copy' in kwargs and kwargs['copy']:
    ...             blob = deepcopy(blob)
    ...         # afterwards set it and call the parents init
    ...         self.blob = blob
    ...         super(NDDataWithBlob, self).__init__(*args, **kwargs)
    ...
    ...     @property
    ...     def blob(self):
    ...         return self._blob
    ...
    ...     @blob.setter
    ...     def blob(self, value):
    ...         self._blob = value
    ...
    ...     def _slice(self, item):
    ...         # slice all normal attributes
    ...         kwargs = super(NDDataWithBlob, self)._slice(item)
    ...         # The arguments for creating a new instance are saved in kwargs
    ...         # so we need to add another keyword "blob" and add the sliced
    ...         # blob
    ...         kwargs['blob'] = self.blob[item]
    ...         return kwargs # these must be returned

    >>> ndd = NDDataWithBlob([1,2,3], blob=[0, 0.2, 0.3])
    >>> ndd2 = ndd[1:3]
    >>> ndd2.blob
    [0.2, 0.3]

If you wanted to keep just the original ``blob`` instead of the sliced ones
you could use ``kwargs['bob'] = self.blob`` and omit the ``[item]``.


Arithmetic on an existing property
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Customizing how an existing property is handled during arithmetic is possible
with some arguments to the function calls like
:meth:`~nddata.nddata.mixins.NDArithmeticMixin.add` but it's possible to
hardcode behaviour too. The actual operation on the attribute (except for
``unit``) is done in a method ``_arithmetic_*`` where ``*`` is the name of the
property.

For example to customize how the ``meta`` will be affected during arithmetics::

    >>> from copy import deepcopy
    >>> class NDDataWithMetaArithmetics(NDData):
    ...
    ...     def _arithmetic_meta(self, operation, operand, handle_mask, **kwds):
    ...         # the function must take the arguments:
    ...         # operation (numpy-ufunc like np.add, np.subtract, ...)
    ...         # operand (the other NDData-like object, already wrapped as NDData)
    ...         # handle_mask (see description for "add")
    ...
    ...         # The meta is dict like but we want the keywords exposure to change
    ...         # Anticipate that one or both might have no meta and take the first one that has
    ...         result_meta = deepcopy(self.meta) if self.meta else deepcopy(operand.meta)
    ...         # Do the operation on the keyword if the keyword exists
    ...         if result_meta and 'exposure' in result_meta:
    ...             result_meta['exposure'] = operation(result_meta['exposure'], operand.data)
    ...         return result_meta # return it

To trigger this method the ``handle_meta`` argument to arithmetic methods can
be anything except ``None`` or ``"first_found"``::

    >>> ndd = NDDataWithMetaArithmetics([1,2,3], meta={'exposure': 10})
    >>> ndd2 = ndd.add(10, handle_meta='')
    >>> ndd2.meta
    {'exposure': 20}

    >>> ndd3 = ndd.multiply(0.5, handle_meta='')
    >>> ndd3.meta
    {'exposure': 5.0}

.. warning::
  To use these internal ``_arithmetic_*`` methods there are some restrictions
  on the attributes when calling the operation:

  - ``mask``: ``handle_mask`` must not be ``None``, ``"ff"`` or ``"first_found"``.
  - ``wcs``: ``compare_wcs`` argument with the same restrictions as mask.
  - ``meta``: ``handle_meta`` argument with the same restrictions as mask.
  - ``uncertainty``: ``propagate_uncertainties`` must be ``None`` or evaluate
    to ``False``. ``arithmetic_uncertainty`` must also accepts different
    arguments: ``operation, operand, result, correlation, **kwargs``


Changing default argument for arithmetic operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the goal is to change the default value of an existing parameter for
arithmetic methods, maybe because explicitly specifying the parameter each
time you're calling an arithmetic operation is too much effort, you can easily
change the default value of existing parameters by changing it in the method
signature of ``_arithmetic``::

    >>> from nddata.nddata import NDData
    >>> import numpy as np

    >>> class NDDDiffAritDefaults(NDData):
    ...     def _arithmetic(self, *args, **kwargs):
    ...         # Changing the default of handle_mask to None
    ...         if 'handle_mask' not in kwargs:
    ...             kwargs['handle_mask'] = None
    ...         # Call the original with the updated kwargs
    ...         return super(NDDDiffAritDefaults, self)._arithmetic(*args, **kwargs)

    >>> ndd1 = NDDDiffAritDefaults(1, mask=False)
    >>> ndd2 = NDDDiffAritDefaults(1, mask=True)
    >>> ndd1.add(ndd2).mask is None  # it will be None
    True

    >>> # But giving other values is still possible:
    >>> ndd1.add(ndd2, handle_mask=np.logical_or).mask
    True

    >>> ndd1.add(ndd2, handle_mask="ff").mask
    False

The parameter controlling how properties are handled are all keyword-only
so using the ``*args, **kwargs`` approach allows one to only alter one default
without needing to care about the positional order of arguments. But using
``def _arithmetic(self, *args, handle_mask=None, **kwargs)`` doesn't work
for python 2.


Arithmetic with an additional property
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This also requires overriding the ``_arithmetic`` method. Suppose we have a
``flags`` attribute again::

    >>> from copy import deepcopy
    >>> import numpy as np

    >>> # The init and attribute is identical to the other blob classes
    >>> class NDDataWithBlob(NDData):
    ...     def __init__(self, *args, **kwargs):
    ...         # We allowed args and kwargs so find out where the data is.
    ...         data = args[0] if args else kwargs['data']
    ...
    ...         # There are three ways to get blob:
    ...         # 1.) explicitly given if they are given they should be used.
    ...         blob = kwargs.pop('blob', None)
    ...         if blob is None:
    ...             # 2.) another NDData - maybe with blob
    ...             if isinstance(data, NDDataBase):
    ...                 blob = getattr(data, 'blob', None)
    ...             # 3.) implements the NDData interface, maybe returns blob
    ...             elif hasattr(data, '__astropy_nddata__'):
    ...                 blob = data.__astropy_nddata__().get('blob', None)
    ...
    ...         # make sure the flags are copied if copy is set.
    ...         if 'copy' in kwargs and kwargs['copy']:
    ...             blob = deepcopy(blob)
    ...         # afterwards set it and call the parents init
    ...         self.blob = blob
    ...         super(NDDataWithBlob, self).__init__(*args, **kwargs)
    ...
    ...     @property
    ...     def blob(self):
    ...         return self._blob
    ...
    ...     @blob.setter
    ...     def blob(self, value):
    ...         self._blob = value
    ...
    ...     def _arithmetic(self, operation, operand, *args, **kwargs):
    ...         # take all args and kwargs to allow arithmetic on the other properties
    ...         # to work like before.
    ...
    ...         # do the arithmetics on the blob (pop the relevant kwargs, if any!!!)
    ...         if self.blob is not None and operand.blob is not None:
    ...             result_blob = np.logical_or(self.blob, operand.blob)
    ...             # np.logical_or is just a suggestion you can do what you want
    ...         else:
    ...             if self.blob is not None:
    ...                 result_blob = deepcopy(self.blob)
    ...             else:
    ...                 result_blob = deepcopy(operand.blob)
    ...
    ...         # Let the superclass do all the other attributes note that
    ...         # this returns the result and a dictionary containing other attributes
    ...         result, kwargs = super(NDDataWithBlob, self)._arithmetic(operation, operand, *args, **kwargs)
    ...         # The arguments for creating a new instance are saved in kwargs
    ...         # so we need to add another keyword "blob" and add the processed blob
    ...         kwargs['blob'] = result_blob
    ...         return result, kwargs # these must be returned

    >>> ndd1 = NDDataWithBlob([1,2,3], blob=np.array([1,0,1], dtype=bool))
    >>> ndd2 = NDDataWithBlob([1,2,3], blob=np.array([0,0,1], dtype=bool))
    >>> ndd3 = ndd1.add(ndd2)
    >>> ndd3.blob
    array([ True, False,  True], dtype=bool)

Another arithmetic operation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Adding another possible operations is quite easy provided the ``data`` and
``unit`` allow it within the framework of `~astropy.units.Quantity`.

For example adding a power function::

    >>> from nddata.nddata import NDData
    >>> import numpy as np
    >>> from astropy.utils import sharedmethod

    >>> class NDDataPower(NDData):
    ...     @sharedmethod # sharedmethod to allow it also as classmethod
    ...     def pow(self, operand, operand2=None, **kwargs):
    ...         # the uncertainty doesn't allow propagation so set it to None
    ...         kwargs['propagate_uncertainties'] = None
    ...         # Call the _prepare_then_do_arithmetic function with the
    ...         # numpy.power ufunc.
    ...         return self._prepare_then_do_arithmetic(np.power, operand,
    ...                                                 operand2, **kwargs)

This can be used like the other arithmetic methods like
:meth:`~nddata.nddata.mixins.NDArithmeticMixin.add`. So it works when calling
it on the class or the instance::

    >>> ndd = NDDataPower([1,2,3])

    >>> # using it on the instance with one operand
    >>> ndd.pow(3)
    NDDataPower([ 1,  8, 27])

    >>> # using it on the instance with two operands
    >>> ndd.pow([1,2,3], [3,4,5])
    NDDataPower([  1,  16, 243])

    >>> # or using it as classmethod
    >>> NDDataPower.pow(6, [1,2,3])
    NDDataPower([  6,  36, 216])

To allow propagation also with ``uncertainty`` see subclassing
`~nddata.nddata.meta.NDUncertainty`.

The ``_prepare_then_do_arithmetic`` implements the relevant checks if it was
called on the class or the instance and if one or two operands were given and
converts the operands, if necessary, to the appropriate classes. Overriding
this ``_prepare_then_do_arithmetic`` in subclasses should be avoided if
possible.

.. note::
    This example is fictional since `~nddata.nddata.NDData` already implements
    a :meth:`~nddata.nddata.mixins.NDArithmeticMixin.power` method.

`~nddata.nddata.meta.NDDataMeta`
--------------------------------

The class `~nddata.nddata.meta.NDDataMeta` is a metaclass -- when subclassing
it, all properties of `~nddata.nddata.meta.NDDataMeta` *must* be overriden in
the subclass.

Subclassing from `~nddata.nddata.meta.NDDataMeta` gives you complete
flexibility in how you implement data storage and the other properties. If your
data is stored in a numpy array (or something that behaves like a numpy array),
it may be more straightforward to subclass `~nddata.nddata.NDDataBase` instead
of `~nddata.nddata.meta.NDDataMeta`.

Implementing the NDDataMeta interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For example to create a readonly container::

    >>> from nddata.nddata.meta import NDDataMeta

    >>> class NDDataReadOnlyNoRestrictions(NDDataMeta):
    ...     def __init__(self, data, unit, mask, uncertainty, meta, wcs, flags):
    ...         self._data = data
    ...         self._unit = unit
    ...         self._mask = mask
    ...         self._uncertainty = uncertainty
    ...         self._meta = meta
    ...         self._wcs = wcs
    ...         self._flags = flags
    ...
    ...     @property
    ...     def data(self):
    ...         return self._data
    ...
    ...     @property
    ...     def unit(self):
    ...         return self._unit
    ...
    ...     @property
    ...     def mask(self):
    ...         return self._mask
    ...
    ...     @property
    ...     def uncertainty(self):
    ...         return self._uncertainty
    ...
    ...     @property
    ...     def meta(self):
    ...         return self._meta
    ...
    ...     @property
    ...     def wcs(self):
    ...         return self._wcs
    ...
    ...     @property
    ...     def flags(self):
    ...         return self._flags

    >>> # A meaningless test to show that creating this class is possible:
    >>> NDDataReadOnlyNoRestrictions(1,2,3,4,5,6,7) is not None
    True

.. note::
  Actually defining an ``__init__`` is not necessary and the properties could
  return arbitary values but the properties **must** be defined.

Subclassing `~nddata.nddata.meta.NDUncertainty`
-----------------------------------------------
.. warning::
    The internal interface of `~nddata.nddata.meta.NDUncertainty` and
    subclasses is experimental and might change in future versions.

Subclasses deriving from `~nddata.nddata.meta.NDUncertainty` need to implement:

- property ``uncertainty_type``, should return a string describing the
  uncertainty for example ``"ivar"`` for inverse variance.

Creating an uncertainty without propagation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`~nddata.nddata.UnknownUncertainty` is a minimal working implementation
without error propagation. So let's create an uncertainty just storing
systematic uncertainties::

    >>> from nddata.nddata.meta import NDUncertainty

    >>> class SystematicUncertainty(NDUncertainty):
    ...     @property
    ...     def uncertainty_type(self):
    ...         return 'systematic'

    >>> SystematicUncertainty([10])
    SystematicUncertainty([10])

Subclassing `~nddata.nddata.StdDevUncertainty`
-----------------------------------------------

Creating an variance uncertainty
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`~nddata.nddata.StdDevUncertainty` already implements propagation based
on gaussian standard deviation so this could be the starting point of an
uncertainty using these propagations:

    >>> from nddata.nddata import StdDevUncertainty
    >>> import numpy as np
    >>> import weakref

    >>> class VarianceUncertainty(StdDevUncertainty):
    ...     @property
    ...     def uncertainty_type(self):
    ...         return 'variance'
    ...
    ...     def _propagate_add(self, other_uncert, *args, **kwargs):
    ...         # Neglect the unit assume that both are Variance uncertainties
    ...         this = StdDevUncertainty(np.sqrt(self.data))
    ...         other = StdDevUncertainty(np.sqrt(other_uncert.data))
    ...
    ...         # We need to set the parent_nddata attribute otherwise it will
    ...         # fail for multiplication and division where the data
    ...         # not only the uncertainty matters.
    ...         this.parent_nddata = weakref.ref(self.parent_nddata)
    ...         other.parent_nddata = weakref.ref(other_uncert.parent_nddata)
    ...
    ...         # Call propagation:
    ...         result = this._propagate_add(other, *args, **kwargs)
    ...
    ...         # Return the square of it
    ...         # Special case: Return is already an Uncertainty
    ...         if isinstance(result, NDUncertainty):
    ...             return np.square(result.data)
    ...         return np.square(result)

    >>> from nddata.nddata import NDData

    >>> ndd1 = NDData([1,2,3], unit='m', uncertainty=VarianceUncertainty([1,4,9]))
    >>> ndd2 = NDData([1,2,3], unit='m', uncertainty=VarianceUncertainty([1,4,9]))
    >>> ndd = ndd1.add(ndd2)
    >>> ndd.uncertainty
    VarianceUncertainty([  2.,   8.,  18.])

this approach certainly works if both are variance uncertainties, but if you
want to allow that VarianceUncertainty also propagates with StdDevUncertainty
you must register the conversion::

    >>> from nddata.nddata import UncertaintyConverter
    >>> def var_to_std(uncertainty):
    ...     data = np.sqrt(uncertainty.data)
    ...     unit = np.sqrt(1 * uncertainty.unit).value if uncertainty.unit else None
    ...     parent_nddata = uncertainty.parent_nddata if uncertainty._parent_nddata else None
    ...     return {'data': data, 'unit': unit, 'parent_nddata': parent_nddata}
    >>> def std_to_var(uncertainty):
    ...     data = np.square(uncertainty.data)
    ...     unit = np.square(1 * uncertainty.unit).value if uncertainty.unit else None
    ...     parent_nddata = uncertainty.parent_nddata if uncertainty._parent_nddata else None
    ...     return {'data': data, 'unit': unit, 'parent_nddata': parent_nddata}
    >>> UncertaintyConverter.register(VarianceUncertainty, StdDevUncertainty,
    ...                               var_to_std, std_to_var)

    >>> ndd1 = NDData([1,2,3], uncertainty=VarianceUncertainty([1,4,9]))
    >>> ndd2 = NDData([1,2,3], uncertainty=StdDevUncertainty([1,2,3]))
    >>> ndd = ndd1.add(ndd2)
    >>> ndd.uncertainty
    VarianceUncertainty([  2.,   8.,  18.])

    >>> ndd = ndd2.add(ndd1)
    >>> ndd.uncertainty
    StdDevUncertainty([ 1.41421356,  2.82842712,  4.24264069])

This converter also allows direct conversion between the types simply by using
the constructor::

    >>> VarianceUncertainty(ndd.uncertainty)
    VarianceUncertainty([  2.,   8.,  18.])

.. note::
    Creating a variance uncertainty like this might require more work to
    include proper treatement of the unit of the uncertainty! And of course
    implementing also the ``_propagate_*`` for subtraction, division and
    multiplication.


General Tipps about subclassing
-------------------------------

Unit conversions
^^^^^^^^^^^^^^^^

`~nddata.nddata.NDData` and it's uncertainty allow for different units. In some
cases you need to convert the units. The fastest way seems to be::

    >>> import astropy.units as u
    >>> data = 100
    >>> unit = u.m
    >>> target_unit = u.cm
    >>> unit.to(target_unit, data)
    10000.0

which is faster than creating a Quantity and converting the unit then::

    >>> (data * unit).to(target_unit).value
    10000.0

the difference can be quite high::

    >>> %timeit unit.to(target_unit, data) # doctest: +SKIP
    10000 loops, best of 3: 28.4 µs per loop
    >>> %timeit (data * unit).to(target_unit).value # doctest: +SKIP
    10000 loops, best of 3: 160 µs per loop

Special casing uncertainty propagation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are cases in which one uncertainty is `None`. **Don't** set it to ``0``
and do the general propagation. The calculation is much faster if you special
case them::

    >>> ndd1 = NDData(np.random.random((1000, 1000)),
    ...               StdDevUncertainty(np.random.random((1000, 1000)))) # doctest: +SKIP
    >>> ndd2_zero = NDData(np.random.random((1000, 1000)),
    ...                    StdDevUncertainty(0)) # doctest: +SKIP
    >>> ndd2_none = NDData(np.random.random((1000, 1000)),
    ...                    StdDevUncertainty(None)) # doctest: +SKIP
    >>> %timeit ndd1.multiply(ndd2_zero) # doctest: +SKIP
    10 loops, best of 3: 74.7 ms per loop
    >>> %timeit ndd1.multiply(ndd2_none) # doctest: +SKIP
    10 loops, best of 3: 28.8 ms per loop

while both give the right result. But the not-special case takes 3 times as
long. The difference get's even bigger if more unit conversions are involved.

Uncertainty propagation formulas
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Always make sure that you don't use a formula that may produce unnecessary
``Inf`` or ``NaN``. Check if another representation of the formula avoids these
problems. Even if it means it gets less efficient. Fast but incorrect results
don't help anybody.

Instead of::

    np.abs(AB)*np.sqrt((dA/A)**2+(dB/B)**2+2*dA/A*dB/B*cor)

you could use::

    np.sqrt((dA*B)**2 + (dB*A)**2 + (2 * cor * ABdAdB))

which yields the same result but avoids division where one element could be
``0``.
