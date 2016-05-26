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

    The ``mask`` attribute will be overridden at the end of the clipping
    functions so you should make sure your initial mask is appropriatly applied
    or it is lost.

    The `~nddata.utils.descriptors.ArrayMask` may be a handy descriptor to
    ensure that the ``mask`` is of appropriate type and taken into account
    before the clipping begins.


Clipping extreme values
-----------------------

Using :meth:`~nddata.nddata.mixins.NDClippingMixin.clip_extrema` allows to mask
a specified number of lowest values and highest values in a
`~nddata.nddata.NDDataBase` instance. This method works in place so you should
not catch the result (it will always be ``None``).

The ``nlow`` and ``nhigh`` parameters indicate how many elements will be
clipped::

    >>> import numpy as np
    >>> from nddata.nddata import NDData
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

.. warning::
    This function can become very slow if the ``data`` is very big and/or the
    ``nlow`` and ``nhigh`` parameters are great. A test run with a
    ``(2000, 2000, 10)`` array of integers with ``nlow=1, nhigh=1, axis=2``
    took half a minute to complete!
