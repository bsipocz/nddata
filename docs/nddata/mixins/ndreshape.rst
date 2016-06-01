.. _nddata_reshape:

Reshaping NDData
================

The mixin `~nddata.nddata.mixins.NDReshapeMixin` behaves similar to
`~nddata.nddata.mixins.NDSlicingMixin` in that it alters the shape of the
saved data, and attributes.


Data
----

The ``data`` is the reference point for
:meth:`~nddata.nddata.mixins.NDReshapeMixin.offset` and when the data is a
scalar or ``None`` it will not allow any padding. The ``pad_width`` allows
several ways of providing the width of the paddings. Assuming a 2D dataset::

    >>> from nddata.nddata import NDData
    >>> import numpy as np
    >>> ndd = NDData(np.ones((3, 3), int), mask=np.zeros((3,3), bool))

Providing an integer will assume this padding width for each axis, before and
after::

    >>> nddo = ndd.offset(1)
    >>> nddo
    NDData([[0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0]])

the ``data`` will always be padded with zeros.

If you specifiy a tuple it is used as padding for each axis. The first element
the width for before and the second for after the current values::

    >>> nddo = ndd.offset((1, 2))
    >>> nddo
    NDData([[0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]])

or explicitly giving it for all dimensions (tuple of tuples)::

    >>> nddo = ndd.offset(((1, 2), (2, 1)))
    >>> nddo
    NDData([[0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]])

Mask
----
If the ``mask`` attribute is not ``None`` it will be tried to pad it with ones.
These new elements contain zero data so having them masked seems like a good
idea::

    >>> ndd = NDData(np.ones((3, 3), int), mask=np.zeros((3,3), bool))
    >>> nddo = ndd.offset(((1, 2), (2, 1)))
    >>> nddo.mask
    array([[ True,  True,  True,  True,  True,  True],
           [ True,  True, False, False, False,  True],
           [ True,  True, False, False, False,  True],
           [ True,  True, False, False, False,  True],
           [ True,  True,  True,  True,  True,  True],
           [ True,  True,  True,  True,  True,  True]], dtype=bool)

the masks follows the same rules as the ``data``. In case it cannot be padded
an information is provided and the original value is kept::

    >>> ndd = NDData(np.ones((3, 3), int), mask=True)
    >>> nddo = ndd.offset(((1, 2), (2, 1)))
    INFO: mask cannot be offsetted. [nddata.nddata.mixins.ndreshape]
    >>> nddo.mask
    True

Flags
-----
The ``flags`` behave identical to the ``mask`` except that they are padded with
zeros::

    >>> ndd = NDData(np.ones((3, 3), int), flags=np.arange(1,10).reshape(3, 3))
    >>> nddo = ndd.offset(((1, 2), (2, 1)))
    >>> nddo.flags
    array([[0, 0, 0, 0, 0, 0],
           [0, 0, 1, 2, 3, 0],
           [0, 0, 4, 5, 6, 0],
           [0, 0, 7, 8, 9, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0]])

WCS
-----
The ``wcs`` will be padded by modifying the ``crpix`` values if it's an
`~astropy.wcs.WCS` object::

    >>> from astropy.wcs import WCS
    >>> w = WCS(naxis=2)
    >>> w.wcs.crpix = [10, 12]

    >>> ndd = NDData(np.ones((3, 3), int), wcs=w)
    >>> nddo = ndd.offset((1, 0))
    >>> nddo.wcs
    WCS Keywords
    <BLANKLINE>
    Number of WCS axes: 2
    CTYPE : ''  ''
    CRVAL : 0.0  0.0
    CRPIX : 11.0  13.0
    PC1_1 PC1_2  : 1.0  0.0
    PC2_1 PC2_2  : 0.0  1.0
    CDELT : 1.0  1.0
    NAXIS    : 0 0

If it isn't the same rules apply as for the ``flags``::

    >>> ndd = NDData(np.ones((3, 3), int), wcs=np.arange(1,10).reshape(3, 3))
    >>> nddo = ndd.offset((1, 0))
    >>> nddo.wcs
    array([[0, 0, 0, 0],
           [0, 1, 2, 3],
           [0, 4, 5, 6],
           [0, 7, 8, 9]])

Uncertainty
-----------
The ``uncertainty`` must implement an ``offset`` method themself. This is
currently implemented for subclasses of `~nddata.nddata.meta.NDUncertainty` if
the ``data`` is a `numpy.ndarray`. Again only an information is printed if it
cannot be offsetted::

    >>> ndd = NDData(np.ones((3, 3), int), uncertainty=np.ones((3,3)))
    INFO: uncertainty should have attribute uncertainty_type. [nddata.utils.descriptors]

    >>> nddo = ndd.offset(2)
    >>> nddo.uncertainty
    UnknownUncertainty([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.],
                        [ 0.,  0.,  1.,  1.,  1.,  0.,  0.],
                        [ 0.,  0.,  1.,  1.,  1.,  0.,  0.],
                        [ 0.,  0.,  1.,  1.,  1.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.]])
