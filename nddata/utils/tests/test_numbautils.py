# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from astropy.tests.helper import pytest

from ..numbautils import interpolate, convolve, _process
from ...deps import OPT_DEPS


@pytest.mark.xfail(not OPT_DEPS['NUMBA'], strict=True, reason="numba required")
def test_processfail():
    with pytest.raises(ValueError):
        _process(1, 1, 1, 'fail')


@pytest.mark.xfail(not OPT_DEPS['NUMBA'], strict=True, reason="numba required")
def test_otherfails():
    # Mask has a different shape
    with pytest.raises(ValueError):
        interpolate([1, 2, 3], [1, 1, 1], [1, 0])
    with pytest.raises(ValueError):
        convolve([1, 2, 3], [1, 1, 1], [1, 0])

    # Kernel has a different number of axis
    with pytest.raises(ValueError):
        interpolate([1, 2, 3], [[1, 1, 1], [1, 1, 1], [1, 1, 1]], [1, 0, 0])
    with pytest.raises(ValueError):
        convolve([1, 2, 3], [[1, 1, 1], [1, 1, 1], [1, 1, 1]], [1, 0, 0])

    # Kernel has an even sized axis
    with pytest.raises(ValueError):
        interpolate([1, 2, 3], [1, 1], [0, 0, 0])
    with pytest.raises(ValueError):
        convolve([1, 2, 3], [1, 1], [0, 0, 0])

    # Array has more than 3 dimensions
    with pytest.raises(ValueError):
        interpolate(np.ones((4, 4, 4, 4)),
                    np.ones((3, 3, 3, 3)),
                    np.zeros((4, 4, 4, 4)))
    with pytest.raises(ValueError):
        convolve(np.ones((4, 4, 4, 4)),
                 np.ones((3, 3, 3, 3)),
                 np.zeros((4, 4, 4, 4)))
