# Licensed under a 3-clause BSD style license - see LICENSE.rst

# TEST_UNICODE_LITERALS

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from astropy.tests.helper import pytest

from ... import NDData
from ..ndstats import SCIPY


def test_fail():
    data = None
    mask = np.random.random((10, 10)) > 0.5
    ndd = NDData(data, mask=mask)

    with pytest.raises(TypeError):
        ndd.stats()


def test_dont_fail():
    data = np.random.random((10, 10))
    mask = np.random.random((10, 10)) > 0.5
    ndd = NDData(data, mask=mask)

    ndd.stats()


def test_contains_astropy_columns():
    data = np.random.random((10, 10))
    mask = np.random.random((10, 10)) > 0.5
    ndd = NDData(data, mask=mask)

    astropy_columns = ['mad', 'biweight_location', 'biweight_midvariance']

    statistics = ndd.stats(astropy=True)

    assert all(opt in statistics.columns for opt in astropy_columns)


@pytest.mark.xfail(not SCIPY, strict=True, reason="scipy_required")
def test_contains_scipy_columns():
    data = np.random.random((10, 10))
    mask = np.random.random((10, 10)) > 0.5
    ndd = NDData(data, mask=mask)

    scipy_columns = ['skew', 'kurtosis']

    statistics = ndd.stats(scipy=True)

    assert all(opt in statistics.columns for opt in scipy_columns)

    statistics = ndd.stats(scipy=True, astropy=True)

    assert all(opt in statistics.columns for opt in scipy_columns)
