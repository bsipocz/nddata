# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from astropy.tests.helper import pytest

from ..numpyutils import is_numeric_array, expand_multi_dims


NUMERIC = [True, 1, -1, 1.0, 1+1j]
NOT_NUMERIC = [object(), 'string', u'unicode', None]


def test_is_numeric():

    for x in NUMERIC:
        for y in (x, [x], [x] * 2):
            for z in (y, np.array(y)):
                assert is_numeric_array(z) is True

    for x in NOT_NUMERIC:
        for y in (x, [x], [x] * 2):
            for z in (y, np.array(y)):
                assert is_numeric_array(z) is False

    for kind, dtypes in np.sctypes.items():
        if kind != 'others':
            for dtype in dtypes:
                assert is_numeric_array(np.array([0], dtype=dtype)) is True


def test_expand_multidims_fail_nd_input():
    """Cannot expand an array that already has more than one dimension."""
    array = np.arange(4).reshape(2, 2)
    with pytest.raises(ValueError):
        expand_multi_dims(array, axis=0, ndims=3)


def test_expand_multidims_fail_axis_out_of_bounds():
    """The axis must be smaller than specified ndims."""
    array = np.arange(4)
    with pytest.raises(ValueError):
        expand_multi_dims(array, axis=1, ndims=1)
    with pytest.raises(ValueError):
        expand_multi_dims(array, axis=2, ndims=2)
    with pytest.raises(ValueError):
        expand_multi_dims(array, axis=5, ndims=5)


def test_expand_multidims_shortcut():
    """Input 1D and target 1D no change"""
    array = np.arange(4)
    res = expand_multi_dims(array, axis=0, ndims=1)
    assert res.shape == array.shape


def test_expand_multidims_others():
    """Normal cases for expanding multiple dimensions"""
    array = np.arange(4)
    # 2D
    assert expand_multi_dims(array, axis=0, ndims=2).shape == (4, 1)
    assert expand_multi_dims(array, axis=1, ndims=2).shape == (1, 4)
    # 3D
    assert expand_multi_dims(array, axis=0, ndims=3).shape == (4, 1, 1)
    assert expand_multi_dims(array, axis=1, ndims=3).shape == (1, 4, 1)
    assert expand_multi_dims(array, axis=2, ndims=3).shape == (1, 1, 4)
    # 4D
    assert expand_multi_dims(array, axis=0, ndims=4).shape == (4, 1, 1, 1)
    assert expand_multi_dims(array, axis=1, ndims=4).shape == (1, 4, 1, 1)
    assert expand_multi_dims(array, axis=2, ndims=4).shape == (1, 1, 4, 1)
    assert expand_multi_dims(array, axis=3, ndims=4).shape == (1, 1, 1, 4)
