# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from astropy import log

from distutils.version import LooseVersion
if LooseVersion(np.__version__) < LooseVersion('1.9'):
    NUMPY_1_9 = False
else:
    NUMPY_1_9 = True


__all__ = ['is_numeric_array', 'mode', 'mode2']

if not NUMPY_1_9:
    __doctest_skip__ = ['mode2']

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


def mode(data):
    """An alternative to the slow `scipy.stats.mode` function.

    This is not an replacement, it's a fast and specialized function which
    doesn't offer as much options and applying much harder restrictions.

    Parameters
    ----------
    data : `np.ndarray`-like
        The data in which to look for the mode. The data will be rounded to
        integer and raveled to a 1D array before looking for the mode.

    Returns
    -------
    most_common_value : `int`
        The most common value in the ``data`` if two values are equally common
        it returns the lower one.

    occurences : `int`
        The number of occurences of the most common value.

    Examples
    --------
    If you have a one dimensional data input containing integer you will get
    the same result as with `scipy.stats.mode`::

        >>> from nddata.utils.numpyutils import mode
        >>> mode([1,1,2,3,4,5])
        (1, 2)

    Which first element is the most common value (``1`` in this case) and the
    number of occurences is the second element (``2``).

    If your array contains float values, they will be rounded with
    `numpy.around` and then the mode is calculated::

        >>> mode([0.1, 0.1, 0.2])
        (0, 3)

    Because they were all rounded to 0.
    """
    # Convert to integer numpy array and flatten it if necessary.
    data = np.asarray(data)
    if data.dtype.kind not in 'ui':
        data = data.round()
    if data.ndim > 1:
        data = data.ravel()

    # Determine the set of possible values.
    # It might not be necessary to shift the bins but it's done nevertheless.
    # Important: arange excludes the last value, so add an additional 1 to the
    # stop value.
    bins = np.arange(np.amin(data)-0.5, np.amax(data)+1.5)

    # Determine the occurences of each value with numpy histogram
    occurences, bins = np.histogram(data, bins)

    # Find the FIRST maximum number of occurences since the bins are from small
    # to high it will return the lower one if two values are equally most
    # common.
    idx_max_count = np.argmax(occurences)

    # Return the value (add 0.5 because we shifted the bins) and the number
    # of occurences.
    return int(round(bins[idx_max_count] + 0.5)), occurences[idx_max_count]


def mode2(data, decimals=0):
    """An alternative to :func:`~nddata.utils.numpyutils.mode` which allows \
            a finer handling of the rounding.

    .. warning::
        This function requires NumPy version 1.9 or newer. For older versions
        of Numpy please use :func:`~nddata.utils.numpyutils.mode`.

    Parameters
    ----------
    data : `np.ndarray`-like
        The data in which to look for the mode.

    decimals : `int` or None, optional
        The number of decimal places to round the data before calculating the
        mode. ``None`` means no rounding (this can be extremly slow). Negative
        values are also possible, see `numpy.around`.
        Default is ``0`` (rounding to integer).

    Returns
    -------
    most_common_value : `int`
        The most common value in the ``data`` if two values are equally common
        it returns the lower one.

    occurences : `int`
        The number of occurences of the most common value.

    Examples
    --------
    Different from the :func:`mode` is that you can specify the ``decimals``::

        >>> from nddata.utils.numpyutils import mode2
        >>> mode2([0.1, 0.1, 0.5], decimals=1)
        (0.10000000000000001, 2)

    That this isn't exactly ``0.1`` is because of the precision of floating
    point values.
    """
    if not NUMPY_1_9:
        if decimals != 0:
            log.info('numpy < 1.9 doesn\'t allow "decimals".')
        return mode(data)

    if decimals is not None:
        data = np.around(data, decimals=decimals)

    values, occurences = np.unique(data, return_counts=True)

    idx_max_count = np.argmax(occurences)

    # Return the value (add 0.5 because we shifted the bins) and the number
    # of occurences.
    return values[idx_max_count], occurences[idx_max_count]
