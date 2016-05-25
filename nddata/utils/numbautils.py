# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import math

import numpy as np

from .. import OPT_DEPS

if OPT_DEPS['NUMBA']:
    import numba as nb


__all__ = ['stats_one_pass', 'stats_two_pass']


def stats_one_pass(array):
    """Calculate several statistical properties in one (or two, not sure yet) \
            over the ``array``.

    .. warning::
        This function requires that `Numba <http://numba.pydata.org/>`_ is
        installed.

    Parameters
    ----------
    array : `numpy.ndarray`-like
        The array for which the statistics should be computed.

    Returns
    -------
    elements : `int`
        The number of elements in the array.

    sum : number
        The sum of all elements.

    mean : number
        The sum divided by the number of elements.

    minimum : number
        The smallest value in the array.

    maximum : number
        The biggest value in the array.

    stddev : number
        The standard deviation assuming 0 degrees of freedom.

    variance : number
        The variance assuming 0 degrees of freedom.
    """
    # Convert it to a plain numpy array
    array = np.asarray(array)
    # Special cases: array only has one element - we could also return a lot
    # of trivial values but it's more likely someone did this by accident.
    # Therefore I let him know!
    if array.size == 1:
        return TypeError('cannot determine statistics with only one element.')
    # Numba requires much explicit loops and since we don't allow an axis
    # argument we can ravel it if it's multidimensional
    if array.ndim > 1:
        array = array.ravel()
    # and return the statistics.
    # TODO: Maybe make this a dictionary otherwise one needs to count which
    # value is which. :-)
    return _numba_stats_one_pass(array)


if OPT_DEPS['NUMBA']:

    @nb.njit
    def _numba_stats_one_pass(array):
        # Initialize the values
        cur_min = array[0]            # Minimum value
        cur_max = array[0]            # Maximum value
        cur_sum = 0.                  # Sum of the elements processed
        cur_rmean = array[0]          # Corrected mean for variance computation
        cur_rsum = 0.                 # Corrected sum for variance computation
        cur_rsum2 = 0.                # Corrected sum squared.

        # Go through the array
        for x in range(array.size):
            # Get the current array elements to avoid indexing always.
            # This doesn't make a performance difference but it makes the code
            # more readable.
            val = array[x]

            # Compare the value to the minimum and maximum and replace those if
            # necessary. This can be done with "elif" since there is no chance
            # it is the new minumum AND maximum.
            if val < cur_min:
                cur_min = val
            elif val > cur_max:
                cur_max = val
            cur_sum += val

            # Use the formula for variance that avoids a catastropic
            # cancellation:
            # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
            diff = val - cur_rmean
            cur_rsum += diff
            cur_rsum2 += diff * diff

        # Determine the final values
        elements = array.size
        sum = cur_sum
        mean = sum / elements
        minimum = cur_min
        maximum = cur_max
        # For variance formula see above wikipedia link. The standard deviation
        # is just the square root of the variance.
        variance = (cur_rsum2 - (cur_rsum * cur_rsum) / elements) / elements
        stddev = math.sqrt(variance)

        return elements, sum, mean, minimum, maximum, stddev, variance

else:

    def _numba_stats_one_pass(array):
        raise ImportError('you need numba to use this function.')
