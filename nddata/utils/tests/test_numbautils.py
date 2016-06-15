# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from astropy.tests.helper import pytest

from ..numbautils import (interpolate, convolve, _process,
                          convolve_median, interpolate_median)
from ...deps import OPT_DEPS


@pytest.mark.xfail(not OPT_DEPS['NUMBA'], strict=True, reason="numba required")
def test_processfail():
    with pytest.raises(ValueError):
        _process(1, 1, 1, 'fail')


@pytest.mark.xfail(not OPT_DEPS['NUMBA'], strict=True, reason="numba required")
@pytest.mark.parametrize(('func'), [interpolate, convolve,
                                    convolve_median, interpolate_median])
def test_otherfails(func):
    # Mask has a different shape
    with pytest.raises(ValueError):
        func([1, 2, 3], [1, 1, 1], [1, 0])

    # Kernel has a different number of axis
    with pytest.raises(ValueError):
        func([1, 2, 3], [[1, 1, 1], [1, 1, 1], [1, 1, 1]], [1, 0, 0])

    # Kernel has an even sized axis
    with pytest.raises(ValueError):
        func([1, 2, 3], [1, 1], [0, 0, 0])

    # Array has more than 3 dimensions
    with pytest.raises(ValueError):
        func(np.ones((4, 4, 4, 4)), np.ones((3, 3, 3, 3)),
             np.zeros((4, 4, 4, 4)))


@pytest.mark.xfail(not OPT_DEPS['NUMBA'], strict=True, reason="numba required")
@pytest.mark.parametrize(('func'), [interpolate, convolve,
                                    convolve_median, interpolate_median])
def test_no_values(func):
    # 1D
    data = np.ones(20)
    mask = np.zeros(20)
    mask[4:7] = 1
    kernel = np.ones(3)
    assert np.isnan(func(data, kernel, mask)[5])
    # 2D
    data = np.ones((10, 10))
    mask = np.zeros((10, 10))
    mask[4:7, 4:7] = 1
    kernel = np.ones((3, 3))
    assert np.isnan(func(data, kernel, mask)[5, 5])
    # 3D
    data = np.ones((10, 10, 10))
    mask = np.zeros((10, 10, 10))
    mask[4:7, 4:7, 4:7] = 1
    kernel = np.ones((3, 3, 3))
    assert np.isnan(func(data, kernel, mask)[5, 5, 5])
