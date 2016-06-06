# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from numpy.testing import assert_array_equal

from astropy.tests.helper import pytest

from ..numpyutils import (create_slices, is_numeric_array, expand_multi_dims,
                          pad)


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


def test_pad():
    with pytest.raises(ValueError):
        pad(2, (1, 1), 'constant', 0)
    with pytest.raises(ValueError):
        pad([2], (1, 1), 'constants', 0)
    with pytest.raises(ValueError):
        pad([2], (1, 1), 'constant', (2, 2))

    # Normal case offsets tuple of tuple
    assert_array_equal(pad([2], ((1, 1), ), 'constant', 0),  [0, 2, 0])
    # Special case for 1d arrays
    assert_array_equal(pad([2], (1, 1), 'constant', 0),  [0, 2, 0])
    assert_array_equal(pad([2], (1, 1), 'constant', np.int32(4)),  [4, 2, 4])
    assert_array_equal(pad([2], (1, 1), 'constant', 10), [10, 2, 10])
    assert_array_equal(pad([2], (1, 2), 'constant', 10), [10, 2, 10, 10])
    assert_array_equal(pad([2], (2, 1), 'constant', 10), [10, 10, 2, 10])
    assert_array_equal(pad([2], (0, 0), 'constant', 10), [2])
    assert_array_equal(pad([2], (0, 1), 'constant', 10), [2, 10])
    assert_array_equal(pad([2], (1, 0), 'constant', 10), [10, 2])


def test_create_slices():
    # Failures
    # Unknown origin parameter
    with pytest.raises(ValueError):
        create_slices(1, 1, 'blub')
    # Incompatible position and shape
    with pytest.raises(TypeError):
        create_slices((1, ), 1, 'start')
    with pytest.raises(TypeError):
        create_slices(1, (1, ), 'start')

    # Correctness tests with scalars
    assert create_slices(1, 1, 'start') == (slice(1, 2, None), )
    assert create_slices(1, 1, 'end') == (slice(1, 2, None), )
    assert create_slices(1, 1, 'center') == (slice(1, 2, None), )

    assert create_slices(1, 2, 'start') == (slice(1, 3, None), )
    assert create_slices(1, 2, 'end') == (slice(0, 2, None), )
    assert create_slices(1, 2, 'center') == (slice(0, 2, None), )

    assert create_slices(2, 3, 'start') == (slice(2, 5, None), )
    assert create_slices(2, 3, 'end') == (slice(0, 3, None), )
    assert create_slices(2, 3, 'center') == (slice(1, 4, None), )

    # And with tuples
    assert create_slices((2, 2), (3, 3), 'start') == (slice(2, 5, None),
                                                      slice(2, 5, None))
    assert create_slices((2, 2), (3, 3), 'end') == (slice(0, 3, None),
                                                    slice(0, 3, None))
    assert create_slices((2, 2), (3, 3), 'center') == (slice(1, 4, None),
                                                       slice(1, 4, None))

    # and with an array as shape parameter
    assert create_slices(1, np.ones(3)) == (slice(1, 4, None), )
    assert create_slices((1, ), np.ones(3)) == (slice(1, 4, None), )
    assert create_slices((1, 1), np.ones((3, 3))) == (slice(1, 4, None),
                                                      slice(1, 4, None))
