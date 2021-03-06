.. _nddata_arithmetic:

NDData Arithmetic
=================

Introduction
------------

`~nddata.nddata.NDData` implements the following arithmetic operations:

- addition: :meth:`~nddata.nddata.mixins.NDArithmeticMixin.add`
- subtraction: :meth:`~nddata.nddata.mixins.NDArithmeticMixin.subtract`
- multiplication: :meth:`~nddata.nddata.mixins.NDArithmeticMixin.multiply`
- division: :meth:`~nddata.nddata.mixins.NDArithmeticMixin.divide`
- exponentation: :meth:`~nddata.nddata.mixins.NDArithmeticMixin.power`

.. warning::
    If you are using objects with ``unit`` then you need to be careful with
    :meth:`~nddata.nddata.mixins.NDArithmeticMixin.power` especially if also
    ``uncertainty`` is used.

Using basic arithmetic methods
------------------------------

Using the standard arithmetic methods requires that the first operand
is an `~nddata.nddata.NDData` instance

    >>> from nddata.nddata import NDData
    >>> import numpy as np
    >>> ndd1 = NDData([1, 2, 3, 4])

while the requirement for the second operand is simply: It must be convertible
to the first operand. It can be a number::

    >>> ndd1.add(3)
    NDData([4, 5, 6, 7])

or a `list`::

    >>> ndd1.subtract([1,1,1,1])
    NDData([0, 1, 2, 3])

a `numpy.ndarray`::

    >>> ndd1.multiply(np.arange(4, 8))
    NDData([ 4, 10, 18, 28])
    >>> ndd1.divide(np.arange(1,13).reshape(3,4))  # a 3 x 4 numpy array
    NDData([[ 1.        ,  1.        ,  1.        ,  1.        ],
               [ 0.2       ,  0.33333333,  0.42857143,  0.5       ],
               [ 0.11111111,  0.2       ,  0.27272727,  0.33333333]])

here broadcasting takes care of the different dimensions. Also several other
classes are possible.

Using arithmetic classmethods
-----------------------------

Here both operands don't need to be `~nddata.nddata.NDData`-like::

    >>> NDData.add(1, 3)
    NDData(4)

or to wrap the result of an arithmetic operation between two Quantities::

    >>> import astropy.units as u
    >>> ndd = NDData.multiply([1,2] * u.m, [10, 20] * u.cm)
    >>> ndd
    NDData([ 10.,  40.])
    >>> ndd.unit
    Unit("cm m")

or taking the inverse of a `~nddata.nddata.NDData` object::

    >>> NDData.divide(1, ndd1)
    NDData([ 1.        ,  0.5       ,  0.33333333,  0.25      ])


Possible operands
^^^^^^^^^^^^^^^^^

The possible types of input for operands are:

+ scalars of any type
+ lists containing numbers (or nested lists)
+ numpy arrays
+ numpy masked arrays
+ astropy quantities
+ other nddata classes or subclasses

Advanced options
----------------

The normal python operators ``+``, ``-``, ... are not implemented because
the methods provide several options how to process the additional attributes.

.. note::
    if you are interested in the experimental functionality including pythons
    operators, have a look at `~nddata.nddata.mixins.NDArithmeticPyOpsMixin`
    and `~nddata.nddata.ContextArithmeticDefaults`.

data, unit
^^^^^^^^^^

For ``data`` and ``unit`` there are no parameters. Every arithmetic
operation lets the `~astropy.units.Quantity`-framework evaluate the result
or fail and abort the operation.

Adding two NDData objects with the same unit works::

    >>> ndd1 = NDData([1,2,3,4,5], unit='m')
    >>> ndd2 = NDData([100,150,200,50,500], unit='m')

    >>> ndd = ndd1.add(ndd2)
    >>> ndd.data
    array([ 101.,  152.,  203.,   54.,  505.])
    >>> ndd.unit
    Unit("m")

Adding two NDData objects with compatible units also works::

    >>> ndd1.unit = 'pc'
    >>> ndd2.unit = 'lyr'

    >>> ndd = ndd1.subtract(ndd2)
    >>> ndd.data
    array([ -29.66013938,  -43.99020907,  -58.32027876,  -11.33006969,
           -148.30069689])
    >>> ndd.unit
    Unit("pc")

this will keep by default the unit of the first operand. However units will
not be decomposed during division::

    >>> ndd = ndd2.divide(ndd1)
    >>> ndd.data
    array([ 100.        ,   75.        ,   66.66666667,   12.5       ,  100.        ])
    >>> ndd.unit
    Unit("lyr / pc")

mask
^^^^

The ``handle_mask`` parameter for the arithmetic operations implements what the
resulting mask will be. There are several options.

- ``None``, the result will have no ``mask``::

      >>> ndd1 = NDData(1, mask=True)
      >>> ndd2 = NDData(1, mask=False)
      >>> ndd1.add(ndd2, handle_mask=None).mask is None
      True

- ``"first_found"`` or ``"ff"``, the result will have the mask of the first
  operand or if that is None the mask of the second operand::

      >>> ndd1 = NDData(1, mask=True)
      >>> ndd2 = NDData(1, mask=False)
      >>> ndd1.add(ndd2, handle_mask="first_found").mask
      True
      >>> ndd3 = NDData(1)
      >>> ndd3.add(ndd2, handle_mask="first_found").mask
      False

- a function (or an arbitary callable) that takes at least two arguments.
  For example `numpy.logical_or` is the default::

      >>> ndd1 = NDData(1, mask=np.array([True, False, True, False]))
      >>> ndd2 = NDData(1, mask=np.array([True, False, False, True]))
      >>> ndd1.add(ndd2).mask
      array([ True, False,  True,  True], dtype=bool)

  This defaults to ``"first_found"`` in case only one ``mask`` is not None::

      >>> ndd1 = NDData(1)
      >>> ndd2 = NDData(1, mask=np.array([True, False, False, True]))
      >>> ndd1.add(ndd2).mask
      array([ True, False, False,  True], dtype=bool)

  Custom functions are also possible::

      >>> def take_alternating_values(mask1, mask2, start=0):
      ...     result = np.zeros(mask1.shape, dtype=np.bool)
      ...     result[start::2] = mask1[start::2]
      ...     result[start+1::2] = mask2[start+1::2]
      ...     return result

  This function is obviously non-sense but let's see how it performs::

      >>> ndd1 = NDData(1, mask=np.array([True, False, True, False]))
      >>> ndd2 = NDData(1, mask=np.array([True, False, False, True]))
      >>> ndd1.add(ndd2, handle_mask=take_alternating_values).mask
      array([ True, False,  True,  True], dtype=bool)

  and additional parameters can be given by prefixing them with ``mask_``
  (which will be stripped before passing it to the function)::

      >>> ndd1.add(ndd2, handle_mask=take_alternating_values, mask_start=1).mask
      array([False, False, False, False], dtype=bool)
      >>> ndd1.add(ndd2, handle_mask=take_alternating_values, mask_start=2).mask
      array([False, False,  True,  True], dtype=bool)

flags
^^^^^

The ``handle_flags`` parameter for the arithmetic operations implements what
the resulting flags will be. There are several options.

- ``None``, the result will have no ``flags``.::

      >>> ndd1 = NDData(1, flags=True)
      >>> ndd2 = NDData(1, flags=False)
      >>> print(ndd1.add(ndd2, handle_flags=None).flags)
      None

  This is the defaults so no need to specify it::

      >>> print(ndd1.add(ndd2).flags)
      None

- ``"first_found"`` or ``"ff"``, the result will have the flags of the first
  operand or if that is None the flags of the second operand::

      >>> ndd1 = NDData(1, flags=True)
      >>> ndd2 = NDData(1, flags=False)
      >>> ndd1.add(ndd2, handle_flags="first_found").flags
      True
      >>> ndd3 = NDData(1)
      >>> ndd3.add(ndd2, handle_flags="first_found").flags
      False

- a function (or an arbitary callable) that takes at least two arguments.
  For example if the ``flags`` resemble a bitmask one can use
  `numpy.bitwise_or`::

      >>> ndd1 = NDData(1, flags=np.array([0, 1, 2, 3]))
      >>> ndd2 = NDData(1, flags=np.array([1, 0, 1, 0]))
      >>> print(ndd1.add(ndd2, handle_flags=np.bitwise_or).flags)
      [1 1 3 3]

  This requires that both flags are not ``None`` otherwise it will fail. but
  a custom functions could be the solution::

      >>> def bitwise_or(flags1, flags2, logical=False):
      ...     if flags1 is None:
      ...         return flags2
      ...     elif flags2 is None:
      ...         return flags1
      ...
      ...     if logical:
      ...         result = np.logical_or(flags1, flags2)
      ...     else:
      ...         result = np.bitwise_or(flags1, flags2)
      ...
      ...     return result

  This function is now also works if one operand has no flags::

      >>> ndd1 = NDData(1, flags=np.array([0, 1, 2, 3]))
      >>> ndd2 = NDData(1)
      >>> ndd1.add(ndd2, handle_flags=bitwise_or).flags
      array([0, 1, 2, 3])

  Additional parameters can be passed to this function as well, just prepend
  a ``flags_`` to the parameter name::

      >>> ndd1 = NDData(1, flags=np.array([0, 1, 2, 3]))
      >>> ndd2 = NDData(1, flags=np.array([0, 0, 1, 0]))
      >>> ndd1.add(ndd2, handle_flags=bitwise_or, flags_logical=True).flags
      array([False,  True,  True,  True], dtype=bool)

meta
^^^^

The ``handle_meta`` parameter for the arithmetic operations implements what the
resulting meta will be. The options are the same as for the ``mask``:

- If ``None`` the resulting ``meta`` will be an empty `collections.OrderedDict`.

      >>> ndd1 = NDData(1, meta={'object': 'sun'})
      >>> ndd2 = NDData(1, meta={'object': 'moon'})
      >>> ndd1.add(ndd2, handle_meta=None).meta
      OrderedDict()

  For ``meta`` this is the default so you don't need to pass it in this case::

      >>> ndd1.add(ndd2).meta
      OrderedDict()

- If ``"first_found"`` or ``"ff"`` the resulting meta will be the meta of the
  first operand or if that contains no keys the meta of the second operand is
  taken.

      >>> ndd1 = NDData(1, meta={'object': 'sun'})
      >>> ndd2 = NDData(1, meta={'object': 'moon'})
      >>> ndd1.add(ndd2, handle_meta='ff').meta
      {'object': 'sun'}

- If it's a ``callable`` it must take at least two arguments. Both ``meta``
  attributes will be passed to this function (even if one or both of them are
  empty) and the callable evaluates the result's meta. For example just a
  function that merges these two::

      >>> # It's expected with arithmetics that the result is not a reference,
      >>> # so we need to copy
      >>> from copy import deepcopy

      >>> def combine_meta(meta1, meta2):
      ...     if not meta1:
      ...         return deepcopy(meta2)
      ...     elif not meta2:
      ...         return deepcopy(meta1)
      ...     else:
      ...         meta_final = deepcopy(meta1)
      ...         meta_final.update(meta2)
      ...         return meta_final

      >>> ndd1 = NDData(1, meta={'time': 'today'})
      >>> ndd2 = NDData(1, meta={'object': 'moon'})
      >>> ndd1.subtract(ndd2, handle_meta=combine_meta).meta # doctest: +SKIP
      {'object': 'moon', 'time': 'today'}

  Here again additional arguments for the function can be passed in using
  the prefix ``meta_`` (which will be stripped away before passing it to this)
  function. See the description for the mask-attribute for further details.

wcs
^^^

The ``compare_wcs`` argument will determine what the result's ``wcs`` will be
or if the operation should be forbidden. The possible values are identical to
``mask`` and ``meta``:

- If ``None`` the resulting ``wcs`` will be an empty ``None``.

      >>> ndd1 = NDData(1, wcs=0)
      >>> ndd2 = NDData(1, wcs=1)
      >>> ndd1.add(ndd2, compare_wcs=None).wcs is None
      True

- If ``"first_found"`` or ``"ff"`` the resulting wcs will be the wcs of the
  first operand or if that is None the meta of the second operand is
  taken.

      >>> ndd1 = NDData(1, wcs=1)
      >>> ndd2 = NDData(1, wcs=0)
      >>> ndd1.add(ndd2, compare_wcs='ff').wcs
      1

- If it's a ``callable`` it must take at least two arguments. Both ``wcs``
  attributes will be passed to this function (even if one or both of them are
  None) and the callable should return ``True`` if these wcs are identical
  (enough) to allow the arithmetic operation or ``False`` if the arithmetic
  operation should be aborted with a ``ValueError``. If ``True`` the ``wcs``
  are identical and the first one is used for the result::

      >>> def compare_wcs_scalar(wcs1, wcs2, allowed_deviation=0.1):
      ...     if wcs1 is None and wcs2 is None:
      ...         return True  # both have no WCS so they are identical
      ...     if wcs1 is None or wcs2 is None:
      ...         return False  # one has WCS, the other doesn't not possible
      ...     else:
      ...         return abs(wcs1 - wcs2) < allowed_deviation

      >>> ndd1 = NDData(1, wcs=1)
      >>> ndd2 = NDData(1, wcs=1)
      >>> ndd1.subtract(ndd2, compare_wcs=compare_wcs_scalar).wcs
      1

  Additional arguments can be passed in prefixing them with ``wcs_`` (this
  prefix will be stripped away before passing it to the function)::

      >>> ndd1 = NDData(1, wcs=1)
      >>> ndd2 = NDData(1, wcs=2)
      >>> ndd1.subtract(ndd2, compare_wcs=compare_wcs_scalar, wcs_allowed_deviation=2).wcs
      1

  If using `~astropy.wcs.WCS` objects a very handy function to use might be::

      >>> from nddata.utils.wcs import wcs_compare

  see :meth:`astropy.wcs.Wcsprm.compare` for the arguments this comparison
  allows.

uncertainty
^^^^^^^^^^^

The ``propagate_uncertainties`` argument can be used to turn the propagation
of uncertainties on or off.

- If ``None`` the result will have no uncertainty::

      >>> from nddata.nddata import StdDevUncertainty
      >>> ndd1 = NDData(1, uncertainty=StdDevUncertainty(0))
      >>> ndd2 = NDData(1, uncertainty=StdDevUncertainty(1))
      >>> ndd1.add(ndd2, propagate_uncertainties=None).uncertainty is None
      True

- If ``False`` the result will have the first found uncertainty.

  .. note::
      Setting ``propagate_uncertainties=False`` is not generally not
      recommended.

- If ``True`` both uncertainties must be ``NDUncertainty`` subclasses that
  implement propagation. This is possible for
  `~nddata.nddata.StdDevUncertainty`::

      >>> ndd1 = NDData(1, uncertainty=StdDevUncertainty([10]))
      >>> ndd2 = NDData(1, uncertainty=StdDevUncertainty([10]))
      >>> ndd1.add(ndd2, propagate_uncertainties=True).uncertainty
      StdDevUncertainty([ 14.14213562])

uncertainty with correlation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If ``propagate_uncertainties`` is ``True`` you can give also an argument
for ``uncertainty_correlation``. `~nddata.nddata.StdDevUncertainty` cannot
keep track of it's correlations by itself but it can evaluate the correct
resulting uncertainty if the correct ``correlation`` is given.

The default (``0``) represents uncorrelated while ``1`` means correlated and
``-1`` anti-correlated. If given a `numpy.ndarray` it should represent the
element-wise correlation coefficient.

For example without correlation subtracting a `~nddata.nddata.NDData`
instance from itself results in a non-zero uncertainty::

    >>> ndd1 = NDData(1, uncertainty=StdDevUncertainty([10]))
    >>> ndd1.subtract(ndd1, propagate_uncertainties=True).uncertainty
    StdDevUncertainty([ 14.14213562])

Given a correlation of ``1`` because they clearly correlate gives the
correct uncertainty of ``0``::

    >>> ndd1 = NDData(1, uncertainty=StdDevUncertainty([10]))
    >>> ndd1.subtract(ndd1, propagate_uncertainties=True,
    ...               uncertainty_correlation=1).uncertainty
    StdDevUncertainty([ 0.])

which would be consistent with the equivalent operation ``ndd1 * 0``::

    >>> ndd1.multiply(0, propagate_uncertainties=True).uncertainty
    StdDevUncertainty([ 0.])

.. warning::
    The user needs to calculate or know the appropriate value or array manually
    and pass it to ``uncertainty_correlation``. The implementation follows
    general first order error propagation formulas, see for example:
    `Wikipedia <https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Example_formulas>`_.

You can also give element-wise correlations::

    >>> ndd1 = NDData([1,1,1,1], uncertainty=StdDevUncertainty([1,1,1,1]))
    >>> ndd2 = NDData([2,2,2,2], uncertainty=StdDevUncertainty([2,2,2,2]))
    >>> ndd1.add(ndd2,uncertainty_correlation=np.array([1,0.5,0,-1])).uncertainty
    StdDevUncertainty([ 3.        ,  2.64575131,  2.23606798,  1.        ])

The correlation ``np.array([1, 0.5, 0, -1])`` would indicate that the first
element is fully correlated, the second element partially correlates while
element 3 is uncorrelated and 4 is anti-correlated.

uncertainty with unit
^^^^^^^^^^^^^^^^^^^^^

`~nddata.nddata.StdDevUncertainty` implements correct error propagation even
if the unit of the data differs from the unit of the uncertainty::

    >>> ndd1 = NDData([10], unit='m', uncertainty=StdDevUncertainty([10], unit='cm'))
    >>> ndd2 = NDData([20], unit='m', uncertainty=StdDevUncertainty([10]))
    >>> ndd1.subtract(ndd2, propagate_uncertainties=True).uncertainty
    StdDevUncertainty([ 1000.04999875])

With the resulting uncertainty having a unit of ``cm``. The uncertainty follows
the convention that the unit will have the unit of the first operand with
``add`` and ``subtract`` and the combined units in case of ``multiply`` and
``divide``.
