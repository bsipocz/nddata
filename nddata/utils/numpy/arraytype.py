# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

__all__ = ['is_numeric']


def is_numeric(array):
    """Checks if the dtype of the array is numeric.

    Booleans, unsigned integer, signed integer, floats and complex are
    considered numeric.

    Parameters
    ----------
    array : `numpy.ndarray`-like
        The array to inspect.

    Returns
    -------
    is_numeric : `bool`
        True if it is a recognized numerical type and False if object or
        string type.
    """
    return np.asarray(array).dtype.kind in {'b', 'u', 'i', 'f', 'c'}
