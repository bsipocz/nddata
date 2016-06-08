.. _nddata_slicing:

Slicing and Indexing NDData
===========================

Introduction
------------

This page only deals with peculiarities applying to
`~nddata.nddata.NDDataBase`-like classes. For a tutorial about slicing/indexing
see the
`python documentation <https://docs.python.org/tutorial/introduction.html#lists>`_
and `numpy documentation <http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html>`_.

.. warning::
    `~nddata.nddata.NDDataBase` and `~nddata.nddata.NDData` enforce almost no
    restrictions on the properties so it might happen that some **valid but
    unusual** combination of properties always results in an IndexError or
    incorrect results. In this case see :ref:`nddata_subclassing` on how to
    customize slicing for a particular property.


Slicing NDData
--------------

Unlike `~nddata.nddata.NDDataBase` the class `~nddata.nddata.NDData`
implements slicing or indexing. The result will be wrapped inside the same
class as the sliced object.

Getting one element::

    >>> import numpy as np
    >>> from nddata.nddata import NDData

    >>> data = np.array([1, 2, 3, 4])
    >>> ndd = NDData(data)
    >>> ndd[1]
    NDData(2)

Getting a sliced portion of the original::

    >>> ndd[1:3]  # Get element 1 (inclusive) to 3 (exclusive)
    NDData([2, 3])

This will return a reference (and as such **not a copy**) of the original
properties so changing a slice will affect the original::

    >>> ndd_sliced = ndd[1:3]
    >>> ndd_sliced.data[0] = 5
    >>> ndd_sliced
    NDData([5, 3])
    >>> ndd
    NDData([1, 5, 3, 4])

except you indexed only one element (for example ``ndd_sliced = ndd[1]``). Then
the element is a scalar and changes will not propagate to the original.

Slicing NDData including attributes
-----------------------------------

In case a ``wcs``, ``mask``, ``flags`` or ``uncertainty`` is present this
attribute will be sliced too::

    >>> from nddata.nddata import StdDevUncertainty
    >>> data = np.array([1, 2, 3, 4])
    >>> mask = data > 2
    >>> uncertainty = StdDevUncertainty(np.sqrt(data))
    >>> wcs = np.ones(4)
    >>> flags = np.zeros(data.shape, dtype=bool)
    >>> ndd = NDData(data, mask=mask, uncertainty=uncertainty, wcs=wcs, flags=flags)
    >>> ndd_sliced = ndd[1:3]

    >>> ndd_sliced.data
    array([2, 3])

    >>> ndd_sliced.mask
    array([False,  True], dtype=bool)

    >>> ndd_sliced.uncertainty
    StdDevUncertainty([ 1.41421356,  1.73205081])

    >>> ndd_sliced.wcs
    array([ 1.,  1.])

    >>> ndd_sliced.flags
    array([False, False], dtype=bool)

but ``unit`` and ``meta`` will be unaffected.

If any of the attributes is set but doesn't implement slicing an info will be
printed and the property will be kept as is::

    >>> data = np.array([1, 2, 3, 4])
    >>> mask = False
    >>> from nddata.nddata import UnknownUncertainty
    >>> uncertainty = UnknownUncertainty(0)
    >>> wcs = {'a': 5}
    >>> flags = False
    >>> ndd = NDData(data, mask=mask, uncertainty=uncertainty, wcs=wcs, flags=flags)
    >>> ndd_sliced = ndd[1:3]
    INFO: uncertainty cannot be sliced. [nddata.nddata.mixins.ndslicing]
    INFO: mask cannot be sliced. [nddata.nddata.mixins.ndslicing]
    INFO: wcs cannot be sliced. [nddata.nddata.mixins.ndslicing]
    INFO: flags cannot be sliced. [nddata.nddata.mixins.ndslicing]

    >>> ndd_sliced.mask
    False

Example: Remove masked data
---------------------------

.. warning::
    If you are using a `~astropy.wcs.WCS` object as ``wcs`` this will **NOT**
    be possible. But you could work around it, i.e. set it to ``None`` before
    slicing.

By convention the ``mask`` attribute indicates if a point is valid or invalid.
So we are able to get all valid data points by slicing with the mask::

    >>> data = np.array([[1,2,3],[4,5,6],[7,8,9]])
    >>> mask = np.array([[0,1,0],[1,1,1],[0,0,1]], dtype=bool)
    >>> uncertainty = StdDevUncertainty(np.sqrt(data))
    >>> ndd = NDData(data, mask=mask, uncertainty=uncertainty)
    >>> # don't forget that ~ or you'll get the invalid points
    >>> ndd_sliced = ndd[~ndd.mask]
    >>> ndd_sliced
    NDData([1, 3, 7, 8])

    >>> ndd_sliced.mask
    array([False, False, False, False], dtype=bool)

    >>> ndd_sliced.uncertainty
    StdDevUncertainty([ 1.        ,  1.73205081,  2.64575131,  2.82842712])

or all invalid points::

    >>> ndd_sliced = ndd[ndd.mask] # without the ~ now!
    >>> ndd_sliced
    NDData([2, 4, 5, 6, 9])

    >>> ndd_sliced.mask
    array([ True,  True,  True,  True,  True], dtype=bool)

    >>> ndd_sliced.uncertainty
    StdDevUncertainty([ 1.41421356,  2.        ,  2.23606798,  2.44948974,  3.        ])

.. note::
    The result of this kind of indexing (boolean indexing) will always be
    one-dimensional!


Cutouts
-------

Creating cutouts from an `~nddata.nddata.NDData` instance by specifying a
position and a final shape is possible using:

- :meth:`~nddata.nddata.mixins.NDSlicingMixin.slice` based on grid coordinates.
- :meth:`~nddata.nddata.mixins.NDSlicingMixin.slice_cutout` based on grid and
  wcs coordinates. Requires the ``wcs`` to be an `~astropy.wcs.WCS` instance.

The ``slice`` method provides a subset of the ``slice_cutout`` functionality
but it can be used even if no ``wcs`` attribute is set. Otherwise both methods
share the same parameters.

    >>> from astropy.wcs import WCS
    >>> wcs = WCS(naxis=1)
    >>> ndd = NDData(np.arange(100), wcs=wcs)

There are different options to specify what the position represents:

- ``"center"``: Position should be (if possible) the center of the cutout
- ``"start"``: Position is the first element of the cutout
- ``"end"``: Position is the last element of the cutout

    >>> ndd.slice([10], [10], 'start')
    NDData([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])

    >>> ndd.slice_cutout([10], [10], 'start')  # cutout behaves the same
    NDData([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])

``origin="center"`` will put the position in the center for odd sized shapes
but place it one to the right of the middle in case the shape is even::

    >>> ndd.slice([10], [10], 'center')
    NDData([ 5,  6,  7,  8,  9, 10, 11, 12, 13, 14])

    >>> ndd.slice([10], [11], 'center')
    NDData([ 5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15])

``origin="end"`` will include the end-point in the cutout. This differs
slightly from regular ``Python`` and ``NumPy`` slicing which do **not**
include the stop point::

    >>> ndd.slice([10], [10], 'end')
    NDData([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])

For one dimensional ``data`` one can also omit the `list` and only provide
`int` as position and shape::

    >>> ndd.slice([10], [10])
    NDData([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])

    >>> ndd.slice(10, 10)
    NDData([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])

In case the ``shape`` extends beyond the range of the ``data`` the cutout is
trimmed::

    >>> ndd.slice(2, 11, 'center')
    NDData([0, 1, 2, 3, 4, 5, 6, 7])

    >>> ndd.slice(98, 11, 'center')
    NDData([93, 94, 95, 96, 97, 98, 99])

For multidimensional datasets the position and shape **must** contain no more
elements as the ``data`` has dimensions::

    >>> wcs = WCS(naxis=2)
    >>> ndd = NDData(np.arange(100).reshape(10, 10), wcs=wcs)

    >>> ndd.slice(2, 3, 'center')
    NDData([[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
            [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]])

    >>> ndd.slice([2], [3], 'center')
    NDData([[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
            [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]])

    >>> ndd.slice([2, 2], [3, 3], 'center')
    NDData([[11, 12, 13],
            [21, 22, 23],
            [31, 32, 33]])

But with :meth:`~nddata.nddata.mixins.NDSlicingMixin.slice_cutout` these must
contain exactly as many elements as the ``wcs`` has dimensions::

    >>> ndd.slice_cutout([2, 2], [3, 3], 'center')
    NDData([[11, 12, 13],
            [21, 22, 23],
            [31, 32, 33]])

In case the ``wcs`` is an `~astropy.wcs.WCS` instance one can also specify
coordinates instead of grid points for the ``slice_cutout``::

    >>> wcs = WCS(naxis=2)
    >>> wcs.wcs.crpix = [1, 1]
    >>> wcs.wcs.crval = [0, 400]
    >>> wcs.wcs.cdelt = [1, 10]
    >>> wcs.wcs.cunit = ["deg", "nm"]
    >>> ndd = NDData(np.arange(100).reshape(10, 10), wcs=wcs)

To use coordinates just use `~astropy.units.Quantity` as indices instead of
integer::

    >>> import astropy.units as u
    >>> ndd.slice_cutout([2*u.degree, 400*u.nm], [5, 5], 'start')
    NDData([[20, 21, 22, 23, 24],
            [30, 31, 32, 33, 34],
            [40, 41, 42, 43, 44],
            [50, 51, 52, 53, 54],
            [60, 61, 62, 63, 64]])

you can also specify the ``shape`` as coordinates and the position as normal
indices::

    >>> ndd.slice_cutout([3, 3], [5*u.degree, 20*u.nm], 'start')
    NDData([[33, 34, 35],
            [43, 44, 45],
            [53, 54, 55],
            [63, 64, 65],
            [73, 74, 75],
            [83, 84, 85]])

or even both as coordinates::

    >>> ndd.slice_cutout([2*u.degree, 450*u.nm], [5*u.degree, 20*u.nm], 'center')
    NDData([[ 4,  5,  6],
            [14, 15, 16],
            [24, 25, 26],
            [34, 35, 36],
            [44, 45, 46]])

But normal slicing with grid coordinates still works::

    >>> ndd.slice_cutout([2, 2], [3, 3], 'center')
    NDData([[11, 12, 13],
            [21, 22, 23],
            [31, 32, 33]])

.. note::
    Internally the coordinates are converted but to allow slicing these must be
    converted to integer. The function :func:`numpy.around` is used here. Due
    to the rounding the result might seem to be "off by one".

.. warning::
    The coordinates **MUST** be in the same coordinate system the ``wcs``
    specifies. If the ``position`` is given in degrees it doesn't check if the
    coordinate system matches the one of the ``wcs`` so don't expect correct
    results if the ``wcs`` is in galactic coordinates but the position is given
    in ``ICRS`` coordinates.

Apart from the coordinate system the actual unit of the position or shape is
converted to the correct unit::


    >>> ndd.slice_cutout([120*u.arcmin, 450*u.nm], [5*u.degree, 20*u.nm], 'center')
    NDData([[ 4,  5,  6],
            [14, 15, 16],
            [24, 25, 26],
            [34, 35, 36],
            [44, 45, 46]])

Here the ``arcmin`` was converted to the appropriate ``degree`` before
processing it, similar conversions are possible with the coordinates of the
shape::

    >>> ndd.slice_cutout([120 * u.arcmin, 0.45 * u.um],
    ...                  [5 * 60 * 60 * u.arcsec, 20 / 1e9 * u.m], 'center')
    NDData([[ 4,  5,  6],
            [14, 15, 16],
            [24, 25, 26],
            [34, 35, 36],
            [44, 45, 46]])

.. note::
    The conversion from pixel to world coordinates and world to pixel
    coordinates will allways include **all** distortions.