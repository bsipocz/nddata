.. _nddata_clipping:

Masking functions for NDData
============================

`~nddata.nddata.mixins.NDClippingMixin`

Restrictions
------------

The methods introduced by :class:`~nddata.nddata.mixins.NDClippingMixin`
expect the ``mask`` to be a boolean numpy array with the same shape as the
``data``.

.. note::
    If you are using another ``mask`` you may choose:

    - to ignore this Mixin and it's methods or
    - override the ``_clipping_get_mask`` method to interpret your mask
      according to the requirements.
    - override everything necessary.

    The ``mask`` attribute will be overridden at the end of the clipping
    functions so you should make sure your initial mask is appropriatly applied
    or it is lost.

    The `~nddata.utils.descriptors.ArrayMask` may be a handy descriptor to
    ensure that the ``mask`` is of appropriate type and taken into account
    before the clipping begins.


Clipping not finite values
--------------------------

:meth:`~nddata.nddata.mixins.NDClippingMixin.clip_invalid` will mark any
invalid values like ``NaN`` or ``Inf`` as masked::

    >>> import numpy as np
    >>> from nddata.nddata import NDData

    >>> ndd = NDData([1, np.nan, np.inf, -np.inf, 10])
    >>> ndd.clip_invalid()
    >>> ndd.mask
    array([False,  True,  True,  True, False], dtype=bool)

Any previously masked values will stay masked::

    >>> mask = np.array([1,0,0,0,0], dtype=bool)
    >>> ndd = NDData(ndd.data, mask=mask)
    >>> ndd.clip_invalid()
    >>> ndd.mask
    array([ True,  True,  True,  True, False], dtype=bool)


Clipping extreme values
-----------------------

Using :meth:`~nddata.nddata.mixins.NDClippingMixin.clip_extrema` allows to mask
a specified number of lowest values and highest values in a
`~nddata.nddata.NDDataBase` instance. This method works in place so you should
not catch the result (it will always be ``None``).

The ``nlow`` and ``nhigh`` parameters indicate how many elements will be
clipped::

    >>> data = np.array([11, 19, 17,  9,  3, 14, 17,  3, 19, 13])
    >>> ndd = NDData(data)
    >>> ndd.clip_extrema(nhigh=1)
    >>> ndd.mask
    array([False,  True, False, False, False, False, False, False, False, False], dtype=bool)

As this example shows only the first occurence of the highest value is masked
not the second one, the same holds for the lowest value::

    >>> ndd = NDData(data)
    >>> ndd.clip_extrema(nlow=1)
    >>> ndd.mask
    array([False, False, False, False,  True, False, False, False, False, False], dtype=bool)

The ``nlow`` and ``nhigh`` parameter can be combined and also work on
multidimensional arrays::

    >>> ndd = NDData(data.reshape(2, 5))  # reshaped this time
    >>> ndd.clip_extrema(nhigh=2, nlow=1)
    >>> ndd.mask
    array([[False,  True, False, False,  True],
           [False, False, False,  True, False]], dtype=bool)

but can also be applied along an axis::

    >>> ndd = NDData(data.reshape(2, 5))
    >>> ndd.clip_extrema(nhigh=2, nlow=1, axis=1)
    >>> ndd.mask
    array([[False,  True,  True, False,  True],
           [False,  True,  True,  True, False]], dtype=bool)

which now masked the highest two and the lowest one along each axis, so 6 in
total.

But without parameters the method will not mask any values::

    >>> ndd = NDData(data)
    >>> ndd.clip_extrema()
    >>> print(ndd.mask)
    None

.. warning::
    This function can become very slow if the ``data`` is very big and/or the
    ``nlow`` and ``nhigh`` parameters are great. A test run on a low budget
    computer (2016) using integer data of shape ``(2000, 2000, 10)``
    with the parameters ``nhigh=2, nlow=2, axis=0`` took 2 seconds to complete!

Clipping values outside fixed thresholds
----------------------------------------

It is possible to clip values that are below or above a threshold value using
:meth:`~nddata.nddata.mixins.NDClippingMixin.clip_range`. These values are
taken as absolutes (which is different from
:meth:`~nddata.nddata.mixins.NDClippingMixin.clip_sigma` which interprets
deviations).

One can invalidate values below a value with the parameter ``low`` and values
above a threshold given as ``high``::

    >>> data = np.array([0, 10, 20, 30, 40])
    >>> ndd = NDData(data)
    >>> ndd.clip_range(low=10)
    >>> ndd.mask
    array([ True, False, False, False, False], dtype=bool)

The values 10 wasn't masked since it was **not** truly smaller than 10.
Similarly one can mask the values above a threshold::

    >>> ndd.clip_range(high=25)
    >>> ndd.mask
    array([ True, False, False,  True,  True], dtype=bool)

The first value remained masked because the original mask is taken into
account. ``low`` and ``high`` could also be evaluated in one call::

    >>> ndd = NDData(ndd.data)
    >>> ndd.clip_range(low=10, high=25)
    >>> ndd.mask
    array([ True, False, False,  True,  True], dtype=bool)

But without parameters the method will not mask any values::

    >>> ndd = NDData(ndd.data)
    >>> ndd.clip_range()
    >>> print(ndd.mask)
    None

Clipping values based on deviation
----------------------------------

:meth:`~nddata.nddata.mixins.NDClippingMixin.clip_sigma`

some explanation might come in here ... someday.
