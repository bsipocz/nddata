# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from astropy.extern import six

if six.PY2:  # pragma: no cover
    from future_builtins import zip

__all__ = ['create_slices', 'expand_multi_dims', 'is_numeric_array', 'pad']

# Boolean, unsigned integer, signed integer, float, complex.
_NUMERIC_KINDS = set('buifc')


def is_numeric_array(array):
    """Determine whether the argument has a numeric datatype, when
    converted to a NumPy array.

    Booleans, unsigned integers, signed integers, floats and complex
    numbers are the kinds of numeric datatype.

    Parameters
    ----------
    array : `numpy.ndarray`-like
        The array to check.

    Returns
    -------
    is_numeric : `bool`
        True if the array has a numeric datatype, False if not.

    """
    try:
        return array.dtype.kind in _NUMERIC_KINDS
    except AttributeError:
        return np.asarray(array).dtype.kind in _NUMERIC_KINDS


def create_slices(point, shape, origin='start'):
    """Create `slice` to index an array starting from a specified position and\
            with specified shape.

    Parameters
    ----------
    point : `int`, `tuple` of integers
        The position represents the starting/central/end point (inclusive) of
        the slice. The interpretation of the point is controlled by the
        ``origin`` parameter.

    shape : `int`, `tuple` of integers
        The shape represents the extend of the slice. The ``shape`` can also be
        a `numpy.ndarray` in which case it's shape is used.

        .. note::
            The ``point`` and ``shape`` should contain as many integer as
            the target array has dimensions. In case it is a flat (1D) array
            the parameters don't need to be tuples but can also be single
            integer. **But** both parameters must be the same type and contain
            the same number of elements.

    origin : `str` {"start" | "end" | "center"}, optional
        Defines the interpretation of the ``point`` parameter:

        - ``"start"``, first point included in the slice.
        - ``"end"``, last point included in the slice.
        - ``"center"``, central point of the slice. Odd shapes have as many
          elements before and after the center while even shapes have one more
          element before.

        Default is ``"start"``.

    Returns
    -------
    slices : `tuple` of slices
        The returned object can be used to index (slice) an array and get the
        specified parts of the array.

        .. warning::
            The return is always a **tuple** which cannot be used by most
            Python datastructures like `list`, `str`, ... if you want to index
            these you need to get the appropriate element (most probably the
            first) from the return.

    Raises
    ------
    ValueError
        If the ``origin`` is a not allowed type or string.

    See also
    --------
    nddata.nddata.mixins.NDSlicingMixin.slice

    Examples
    --------
    Given an two dimensional 5x10 array::

        >>> from nddata.utils.numpyutils import create_slices
        >>> import numpy as np
        >>> array = np.arange(50).reshape(5, 10)

    For example to get a 3x3 part of the array centered at 2, 5::

        >>> array = np.arange(50).reshape(5, 10)
        >>> slices = create_slices(point=(2, 5), shape=(3, 4), origin='center')
        >>> array[slices]
        array([[13, 14, 15, 16],
               [23, 24, 25, 26],
               [33, 34, 35, 36]])

    Or a 7 element one-dimensional array ending with index 10::

        >>> array = np.arange(15)
        >>> array[create_slices(point=10, shape=7, origin='end')]
        array([ 4,  5,  6,  7,  8,  9, 10])

    These can also be used to insert a small array into a bigger one using a
    central index::

        >>> array2 = np.arange(5)
        >>> array[create_slices(point=4, shape=array2, origin='center')] =\
 array2
        >>> array
        array([ 0,  1,  0,  1,  2,  3,  4,  7,  8,  9, 10, 11, 12, 13, 14])
    """
    # In case the shape is a numpy array we take it's shape. This allows the
    # user to pass in the array which should be inserted.
    try:
        shape = shape.shape
    except AttributeError:
        pass
    else:
        # If we have a numpy array do a quick check that the point is also a
        # tuple (or iterable). This is a bit annoying but .shape always returns
        # a tuple even if the array is 1D or a scalar.
        try:
            len(point)
        except TypeError:
            point = (point, )

    # Zip the point and shape. We require them to be of equal length or integer
    # if they are integer we need to wrap them into tuples before zipping.
    try:
        zips = zip(point, shape)
    except TypeError:
        zips = zip((point, ), (shape, ))

    # Depending on the origin determine the appropriate slices. Start is
    # "normal" slicing but "end" and "center" require some more calculations.
    if origin == 'start':
        # If we start from the ankor is the starting point the slices can be
        # calculated by adding the shape of each dimension to the starting
        # point.
        return tuple(slice(pos, pos + length) for pos, length in zips)

    elif origin == 'end':
        # If the position is the end value we need to subtract the length from
        # the position to get the starting point. But one also needs to add
        # one to the start and the end because otherwise the end point would
        # not be included.
        # TODO: Document that the end point is included here!!!
        return tuple(slice(pos - length + 1, pos + 1) for pos, length in zips)

    elif origin == 'center':
        # Using the center as ankor is more complicated because we might have
        # even and off shapes. We start by calculating an intermediate
        # generator containing the position and the shapes divided by 2. We
        # use floor division and keep the modulo
        zips = ((pos, length // 2, length % 2) for pos, length in zips)
        # The starting point is always just the position minus the result of
        # the floor division, while the stop is the pos plus the result of the
        # floor division AND the modulo. This ensures that off length shapes
        # have as many elements before and after center while even arrays
        # contain one more element before than after.
        return tuple(slice(pos - half_len, pos + half_len + mod)
                     for pos, half_len, mod in zips)

    else:
        raise ValueError('origin must be one of "start", "stop" or "center".')


def pad(array, offsets, mode='constant', constant_values=0):
    """Alternative to :func:`numpy.pad` but only with ``mode=constant``.

    The :func:`numpy.pad` function is very powerful but very slow for small
    inputs and even if it scales exactly like this function it is approximatly
    a factor of 3 slower. This function serves as fast alternative for the
    constant mode with a scalar fill value and as a compatibility layer for
    ``NumPy 1.6`` which did not have a pad function (and some failures on
    ``NumPy 1.7-1.9``).

    Parameters
    ----------
    array : `numpy.ndarray`-like
        The array to pad. If it's a subclass it will be cast to a plain array.

    offsets : `tuple` of `tuple`
        This should be a tuple containing a tuple for each dimension. The first
        element should be the padding before and the second the padding at the
        end. For example ``((1,2),)`` for padding a one-dimensional array with
        one element at the start and two at the end. The convenience options
        of :func:`numpy.pad` are not integrated.

    mode : `str`
        Should be ``"constant"``. Other modes are avaiable using
        :func:`numpy.pad`.
        Default is ``"constant"``.

    constant_values : number
        The value with which to pad the array. Must be a scalar. The more
        advanced options of `numpy.pad` are not allowed.
        Default is ``0``.

    Returns
    -------
    padded_array : `numpy.ndarray`
        The padded array.

    Examples
    --------

    To pad a one-dimensional array::

        >>> from nddata.utils.numpyutils import pad
        >>> import numpy as np

        >>> pad([1,2,3], (1, 2), 'constant', 0)
        array([0, 1, 2, 3, 0, 0])

    But also arbitary dimensional arrays can be padded::

        >>> pad(np.ones((3,3), int), ((0, 1), (2, 1)), 'constant', 0)
        array([[0, 0, 1, 1, 1, 0],
               [0, 0, 1, 1, 1, 0],
               [0, 0, 1, 1, 1, 0],
               [0, 0, 0, 0, 0, 0]])
    """
    # Convert it an array
    array = np.asarray(array)

    # Scalar values should not be padded.
    if array.shape == ():
        raise ValueError('cannot pad scalars.')

    # This is a compatibility function with much less options and optimized
    # just to do constant padding with one value. If that's not enough use
    # np.pad - even though I had some test failures for numpy 1.7-1.9
    # which could mean that there were some changes recently. This is just a
    # way to deal with my common case and ignore the wide range of np.lin.pad
    # possibilities:
    # TL; DR; Only allow constants and only if the value is a scalar.
    if mode != 'constant':
        raise ValueError('pad function can only use mode=constant')

    # In case a 1d array is given the offsets could be a tuple with two
    # elements or a tuple of a tuple of 2 elements. In case it's the first we
    # wrap it inside another tuple so the following parts can remain the same
    # for 1d and multi-d.
    # In case one just happens to enter a ((1,1),(1,1)) as offset this will
    # break calculating the finalshape because of the extra wrapping. But
    # trying to pad a 1D array in 2 dimensions is kinda wrong.
    if array.ndim == 1:
        if len(offsets) == 2:
            offsets = (offsets, )

    # Calculate the finalshape as tuple by adding the current shape to the
    # sum of offsets in this dimension.
    finalshape = tuple(i + offsets[idx][0] + offsets[idx][1]
                       for idx, i in enumerate(array.shape))

    # unfortunatly np.full is only avaiable until numpy 1.8 as long as 1.7 is
    # supported this cannot work.
    # TODO: Use this as soon as numpy 1.7 isn't supported anymore
    # result = np.full(finalshape, dtype=array.dtype,
    #                  fill_value=constant_values)
    result = np.empty(finalshape, dtype=array.dtype)
    result.fill(constant_values)

    # Calculate the position where to insert the array. This can be done by
    # using the create_slices function with origin="start" (default therefore
    # omitted) and the position is just the first element of the offsets.
    # Unfortunatly this requires an intermediate list comprehension but for the
    # shape we can simply use the original array. The function will extract the
    # shape by itself.
    # Then insert the original array in the new one. This will copy the array!
    pos = create_slices([i[0] for i in offsets], array)
    # Without create_slices this would be:
    # pos = tuple(slice(offsets[dim][0], offsets[dim][0]+array.shape[dim], 1)
    #             for dim in range(array.ndim))
    result[pos] = array
    return result


def expand_multi_dims(array, axis, ndims):
    """Add a variable number empty dimensions to an array.

    Parameters
    ----------
    array : `numpy.ndarray`
        The 1D array which should be reshaped.

    axis : positive `int`
        The dimension along which the array is oriented. This will be the
        dimension which **keeps** the original size of the array.

    ndims : positive `int`
        The total number of dimensions of the returned array.

    Returns
    -------
    reshaped_array : `numpy.ndarray`
        The reshaped array.

    Raises
    ------
    ValueError
        In case the ``array`` has more than one dimension already.

    Notes
    -----
    This function is more general than `numpy.expand_dims` because it can add
    multiple dimensions at once. But on the other hand it is more limited
    because it cannot handle multidimensional input.

    This function's primary purpose is to create an array that broadcasts
    correctly to some other data in case the matching dimension is not the
    last one.

    .. note::
        This functions interpretation differs from `numpy.expand_dims` where
        the ``axis`` parameter indicates where the empyt dimension should be
        appended while `expand_multi_dims` ``axis`` parameter indicates where
        the only **not empty** dimension should be.

    Examples
    --------
    For example you have some weights you want to apply along the first
    dimension but you have an 2D array::

        >>> import numpy as np
        >>> weights = np.array([1,2,3])
        >>> array = np.ones((3, 5))

    these cannot be multiplied directly because broadcasting is along the last
    dimension and these differ (5 for array and 3 for weights) this is where
    this function comes in::

        >>> from nddata.utils.numpyutils import expand_multi_dims
        >>> array * expand_multi_dims(weights, axis=0, ndims=2)
        array([[ 1.,  1.,  1.,  1.,  1.],
               [ 2.,  2.,  2.,  2.,  2.],
               [ 3.,  3.,  3.,  3.,  3.]])

    We specified ``axis=0`` because we wanted the 1D array to be applied the
    against the first axis of the other array and because the final array has
    2 dimensions we specified this number as ``ndims=2``.

    In case of 1D and 2D arrays this is rather trivial but the same method can
    be applied, i.e. to correctly broadcast a 1D array along axis=1 against a
    4D array::

        >>> array = np.ones((2,3,4,5))
        >>> res = array * expand_multi_dims(weights, axis=1, ndims=4)

    I neglected displaying the resulting array (120 elements) here but that no
    ValueError was raised indicates that the broadcasting worked as expected.
    """
    array = np.asarray(array)

    # If the input has more than one dimension we cannot expand it in here.
    if array.ndim != 1:
        raise ValueError('cannot expand multiple dimensions of an array with '
                         'more than one dimension. The array has {0} '
                         'dimensions'.format(array.ndim))

    # The axis must be smaller than the specified final ndims
    if axis >= ndims:
        raise ValueError('the alignment axis ({0}) is out of bounds for a '
                         'desired dimensionality ({1})'.format(axis, ndims))

    # The final number of dimensions is 1 we don't need to do anything because
    # the input is already 1d
    if ndims == 1:
        return array

    # Cast axis and ndims to positive integer.
    axis = int(abs(axis))
    ndims = int(abs(ndims))

    # Create a list containing the final shape. Use 1 for every dimension that
    # is not the specified dimension and the array-size for the the dimension
    # along which the result is oriented.
    shape = [1 if ax != axis else array.size for ax in range(ndims)]

    # Use reshape to this intermediate shape and return the result.
    return array.reshape(*shape)
