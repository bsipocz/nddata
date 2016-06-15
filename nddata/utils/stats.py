# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from astropy import log

from .. import MIN_VERSIONS


__all__ = ['mode']

if not MIN_VERSIONS['NUMPY_1_9']:  # pragma: no cover
    __doctest_skip__ = ['mode']


def mode(data, decimals=0):
    """Calculate the mode (most common value) of an array.

    This function is more limited but also a lot faster than
    :func:`scipy.stats.mode` because it rounds the values to a specified
    decimal place.

    .. note::
        ``SciPy`` version 18 will contain an optimized mode function. It is
        still a bit slower than this one but not by a factor of 100-1000 but
        more a factor of 1.5-2. Which might be worth it considering that the
        :func:`scipy.stats.mode` has a lot more options.

    Parameters
    ----------
    data : `numpy.ndarray`-like
        The data in which to look for the mode.

    decimals : `int` or None, optional
        The number of decimal places to round the data before calculating the
        mode.

        - `None` no rounding (this can be extremly slow).
        - `int` : see :func:`numpy.around`.

        Default is ``0`` (rounding to nearest even integer).

        .. warning::
            This parameter requires NumPy version 1.9 or newer. For earlier
            versions it will assume ``decimals=0``.

    Returns
    -------
    most_common_value : number
        The most common value in the ``data`` if two values are equally common
        it returns the lower one. For NumPy < 1.9 it will return a `float`.

    occurences : `int`
        The number of occurences of the most common value.

    See also
    --------
    scipy.stats.mode

    Examples
    --------
    The ``mode`` will calculate the most common value and it's number of
    occurences::

        >>> from nddata.utils.stats import mode
        >>> import numpy as np

        >>> mode([1, 1, 2, 3, 4, 5])
        (1, 2)

    The first element is the most common value (``1.0`` in this case) and the
    number of occurences is the second element (``2``).

    If two values are equally common it returns the lower one::

        >>> mode([1, 2, 3, 4, 5])
        (1, 1)

    The array is flattened before calculation (similar to
    :func:`scipy.stats.mode` with ``axis=None``)::

        >>> mode([[1, 1], [2, 3], [4, 5]])
        (1, 2)

    If your array contains float values, they will be rounded with
    :func:`numpy.around` and then the mode is calculated::

        >>> mode([0.1, 0.1, 0.2])
        (0.0, 3)

    Because they were all rounded to 0 the most common value is ``0.0``. But
    you can specify the number of decimal places (requires NumPy >= 1.9) to
    customize the precision of the mode::

        >>> mode([0.5, 0.5, 0.7], decimals=1)
        (0.5, 2)

    Here ``0.5`` is the most common value and it was found ``2`` times in the
    provided ``data``. Note that the function will be a lot slower if you
    set the ``decimals`` to a high value or to ``None``.

    .. note::
        If you've used ``mode([0.1, 0.1, 0.2])`` the result would've been
        ``(0.10000000000000001, 2)``. That the value differs from ``0.1`` is
        because NumPy and Python use different ways of displaying data. You
        **do not** lose any precision!

    Notes
    -----
    This function is based on :func:`numpy.unique` with ``return_counts``. For
    NumPy < 1.9 :func:`numpy.histogram` is used. Both cases use
    :func:`numpy.argmax` to determine the most common value.

    The rounding before calculating the ``mode`` serves different purposes:
    The likelihood that multiple floats have the same value is very little so
    calculating the mode on full precision float arrays very likely returns
    just the smallest value (because every value is just found once) or some
    random value, that just happens to be multiple times in the ``data``.
    """
    # If someone calls it from numpy 1.8 and lower drop the decimals and send
    # them to the normal mode.
    if not MIN_VERSIONS['NUMPY_1_9']:  # pragma: no cover
        if decimals != 0:
            log.info('numpy < 1.9 doesn\'t allow giving "decimals".')
        return _mode_fallback_numpy_lt_1_9(data)

    data = np.asarray(data)

    # Round the data IF decimals is given.
    if decimals is not None:
        # We don't need to round if the data is integer (unsigned or signed)
        # and decimals was greater or equal to 0 - in case it's smaller it
        # rounds to 10s - not decimals so we always need to round!
        if decimals >= 0 and data.dtype.kind in 'ui':
            pass
        else:
            data = np.around(data, decimals=decimals)

    # Call np.unique to get the values and the counts
    values, occurences = np.unique(data, return_counts=True)

    # Find where the most occurences are. argmax returns the first found value
    # in case two have the same counts. This is the reason why it returns the
    # smaller value.
    idx_max_count = np.argmax(occurences)

    # Return the value and the counts for that value
    return values[idx_max_count], occurences[idx_max_count]


def _mode_fallback_numpy_lt_1_9(data):  # pragma: no cover
    """NumPy 1.8 and earlier don't have "return_counts" for `numpy.unique` so
    this function provides a fallback with `numpy.histogram`.

    Probably it would not be hard to also allow some decimals here especially
    negative ones but making it accept also "None" would be very hard.

    Better to leave the workaround incomplete without trying to hard and just
    let people upgrade their NumPy.
    """
    data = np.asarray(data)

    # Flatten the array if necessary
    if data.ndim > 1:
        data = data.ravel()

    # If NumPy wouldn't round to the nearest EVEN integer we wouldn't need to
    # round here but to ensure that the result is identical: Round if it's not
    # an integer array.

    if data.dtype.kind not in 'ui':
        data = np.around(data)

    # Determine the set of possible values.
    # It might not be necessary to shift the bins but it's done nevertheless.
    # Important: arange excludes the last value, so add an additional 1 to the
    # stop value. Casting it to integer will round down
    binborders = np.arange(int(np.amin(data))-0.5, int(np.amax(data))+1.5)

    # Determine the occurences of each value with numpy histogram
    occurences, binborders = np.histogram(data, binborders)

    # Find the FIRST maximum number of occurences since the bins are from small
    # to high it will return the lower one if two values are equally most
    # common.
    idx_max_count = np.argmax(occurences)

    # Return the value (add 0.5 because we shifted the bins) and the number
    # of occurences.
    return int(binborders[idx_max_count] + 1), occurences[idx_max_count]
