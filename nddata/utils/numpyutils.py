# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

__all__ = ['is_numeric_array', 'expand_multi_dims']

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
