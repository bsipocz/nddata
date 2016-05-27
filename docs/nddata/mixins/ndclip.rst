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

:meth:`~nddata.nddata.mixins.NDClippingMixin.clip_sigma` is loosly based on
:func:`~astropy.stats.sigma_clip` with one major difference: it works in-place,
so the ``data`` is what is saved in the data attribute of the instance coupled
together with the mask attribute, if present. And the resulting mask is used to
update the current mask of the instance. This also eliminates the ``copy``
parameter from the original function.

But it's usage is otherwise identical, you can specify ``sigma`` and/or
explicit ``sigma_lower`` and ``sigma_upper``::

    >>> ndd1 = NDData([5,5,5,1,5,5,5,5])
    >>> ndd2 = ndd1.copy()
    >>> ndd1.clip_sigma(sigma=3)
    >>> ndd1.mask
    array([False, False, False,  True, False, False, False, False], dtype=bool)

    >>> ndd2.clip_sigma(sigma_lower=3, sigma_upper=3)
    >>> ndd2.mask
    array([False, False, False,  True, False, False, False, False], dtype=bool)

are completly identical. Internally ``sigma`` is used as ``sigma_lower`` if
``sigma_lower`` is not explicitly given and similar for ``sigma_upper``.

The algorithm uses the ``cenfunc`` and ``stdfunc`` callables to determine the
center and the standard deviation. Since these are operating on
`numpy.ma.MaskedArray` they need to be able to handle the mask correctly. The
recommended functions are:

- :func:`numpy.mean` for the mean value as center and :func:`numpy.ma.median`
  for the median. Notice that :func:`numpy.median` (without the ``ma``) doesn't
  work for masked arrays while :func:`numpy.mean` does.

- :func:`numpy.std` for the regular standard deviation as deviation or the
  ``AstroPy`` functions :func:`~astropy.stats.mad_std` for the median absolute
  standard deviation or :func:`~astropy.stats.biweight_midvariance`.

You could of course also write your own function and provide it as parameter.
It only needs to satisfy two conditions: It must reduce the dimensions of the
input by 1 (because internally this dimension is added again) and take an
``axis`` parameter. But the defaults should already work well for most cases.

The clip conditions can be summarized by::

    deviation < (-sigma_lower * stdfunc(deviation))
    deviation > (sigma_upper * stdfunc(deviation))

where the deviation is defined as::

    deviation = data - cenfunc(data [,axis=int])

The :meth:`~nddata.nddata.mixins.NDClippingMixin.clip_sigma` also takes an
``axis`` parameter which indicates along which axis the clipping should be
done. The default is ``None`` which performs it along the whole array but any
axis, provided the data has appropriate dimensions, is possible::

    >>> ndd = NDData([[2,2,2,2,2.1], [30,30,30,30,2], [50,50,50,50,4]])

    >>> ndd1 = ndd.copy()
    >>> ndd1.clip_sigma(sigma=2, axis=None)
    >>> ndd1.mask
    array([[False, False, False, False, False],
           [False, False, False, False, False],
           [False, False, False, False, False]], dtype=bool)

    >>> ndd2 = ndd.copy()
    >>> ndd2.clip_sigma(sigma=2, axis=0)
    >>> ndd2.mask
    array([[False, False, False, False, False],
           [False, False, False, False, False],
           [False, False, False, False,  True]], dtype=bool)

    >>> ndd3 = ndd.copy()
    >>> ndd3.clip_sigma(sigma=2, axis=1)
    >>> ndd3.mask
    array([[False, False, False, False,  True],
           [False, False, False, False,  True],
           [False, False, False, False,  True]], dtype=bool)

and even negative ``axis`` are possible (negative axis are interpreted as
counting from the last axis, so ``-1`` is the last axis, ``-2`` the second last
and so on)::

    >>> ndd4 = ndd.copy()
    >>> ndd3.clip_sigma(sigma=2, axis=-1)
    >>> ndd3.mask
    array([[False, False, False, False,  True],
           [False, False, False, False,  True],
           [False, False, False, False,  True]], dtype=bool)

These results differ from each other depending on the chosen ``axis``.

The remaining parameter ``iters`` controls how many clipping iterations are
done. The default is ``None`` which means that the iterations only stop when
no further value is discarded in the last iteration. If you want to limit the
maximal number of iterations then you can provide a custom value here::

    >>> ndd = NDData([5,5,5,5,5,5,5,5,5,5,7,7,7,9])
    >>> ndd.clip_sigma(sigma=2, iters=1)
    >>> ndd.mask
    array([False, False, False, False, False, False, False, False, False,
           False, False, False, False,  True], dtype=bool)

    >>> ndd.clip_sigma(sigma=2, iters=2)
    >>> ndd.mask
    array([False, False, False, False, False, False, False, False, False,
           False,  True,  True,  True,  True], dtype=bool)
