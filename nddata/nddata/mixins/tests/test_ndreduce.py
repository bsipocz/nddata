# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from astropy.tests.helper import pytest

from ... import NDData
from ...nduncertainty_var import VarianceUncertainty


def test_fail_1d():
    ndd = NDData([1, 2, 3, 4, 5], mask=np.array([0, 1, 0, 1, 0], dtype=bool))
    with pytest.raises(ValueError):
        ndd.reduce_average(axis=0, weights=[1, 1, 1, 1, 1])


def test_2d_simple():
    # Tests that if the array has 2d but one of them is empty it will work
    ndd = NDData([[1, 2, 3, 4, 5]],
                 mask=np.array([[0, 1, 0, 1, 0]], dtype=bool))
    avg = ndd.reduce_average(axis=1, weights=[1, 1, 1, 1, 1])

    assert avg.data[0] == (1 + 3 + 5) / 3
    assert not avg.mask[0]
    assert isinstance(avg.uncertainty, VarianceUncertainty)
    assert_allclose(avg.uncertainty.data[0],
                    ((2*2) + (0*0) + (2*2)) / 9)

    # now let's alter the weights a bit to see if they are interpreted
    # correctly
    ndd = NDData([[1, 2, 3, 4, 5]],
                 mask=np.array([[0, 1, 0, 1, 0]], dtype=bool))
    avg = ndd.reduce_average(axis=1, weights=[2, 1, 1, 1, 1])

    assert avg.data[0] == (1*2 + 3 + 5) / 4  # 2.5
    assert not avg.mask[0]
    assert isinstance(avg.uncertainty, VarianceUncertainty)
    assert_allclose(avg.uncertainty.data[0],
                    ((1.5**2)*2 + (0.5**2) + (2.5**2)) / 16)


def test_2d_complicated():
    ndd = NDData([[3, 2, 1, 1, 4], [2, 2, 2, 2, 2]],
                 mask=np.array([[0, 1, 0, 1, 0], [0, 1, 0, 0, 0]], dtype=bool))
    avg = ndd.reduce_average(axis=0, weights=[1, 1.5])

    assert isinstance(avg.uncertainty, VarianceUncertainty)

    # Multiply the data by the sum of the weights. Makes it easier...
    assert_allclose(avg.data*2.5, [3+2*1.5, 0, 1+2*1.5, 2*2.5, 4+2*1.5])
    #                             [ 2.4   , 0.,1.6    , 2.   , 2.8]]

    # This mask has one final masked item!
    assert_array_equal(avg.mask, [0, 1, 0, 0, 0])

    # Multiply the resulting uncertainty times the sum of the weights squared.
    # so I don't need to divide all resulting values by it.
    assert_allclose(avg.uncertainty.data * 2.5**2,
                    [(3-2.4)**2 + 1.5*(2-2.4)**2,
                     0.,
                     (1-1.6)**2 + 1.5*(2-1.6)**2,
                     0.,
                     (4-2.8)**2 + 1.5*(2-2.8)**2])
    #               [0.096, 0., 0.096, 0., 0.384]

    # The same along axis=1
    avg = ndd.reduce_average(axis=1, weights=[1, 1, 1, 1, 2])

    assert isinstance(avg.uncertainty, VarianceUncertainty)

    assert_allclose(avg.data, [(3*1 + 2*0 + 1*1 + 1*0 + 4*2) / 4,
                               (2*1 + 2*0 + 2*1 + 2*1 + 2*2) / 5])
    #                         [ 2.8,  2. ]

    # This mask has no masked value
    assert_array_equal(avg.mask, [0, 0])

    assert_allclose(avg.uncertainty.data,
                    [((3-3)**2 + (1-3)**2 + 2*(4-3)**2) / 16,
                     0])
    #               [ 0.272,  0.   ]
