# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

__all__ = ['is_numeric_array', 'expand_multi_dims', 'pad']

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


def pad(array, offsets, mode, constant_values):
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

    constant_values : number
        The value with which to pad the array. Must be a scalar. The more
        advanced options of `numpy.pad` are not integrated.

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

    # Calculate the position where to insert the array. This is simply
    # start=offset_before, end=offset_before+original_shape. Then insert the
    # original array in the new one. This will copy the array!
    pos = tuple(slice(offsets[dim][0], offsets[dim][0]+array.shape[dim], 1)
                for dim in range(array.ndim))
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
