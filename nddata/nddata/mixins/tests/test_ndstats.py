# Licensed under a 3-clause BSD style license - see LICENSE.rst

# TEST_UNICODE_LITERALS

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from ... import NDData
from astropy.tests.helper import pytest
from ..ndstats import SCIPY


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


def test_contains_scipy_columns():
    data = np.random.random((10, 10))
    mask = np.random.random((10, 10)) > 0.5
    ndd = NDData(data, mask=mask)

    scipy_columns = ['skew', 'kurtosis']

    statistics = ndd.stats(scipy=True)

    if SCIPY:
        assert all(opt in statistics.columns for opt in scipy_columns)
    else:
        assert all(opt not in statistics.columns for opt in scipy_columns)

    statistics = ndd.stats(scipy=True, astropy=True)

    if SCIPY:
        assert all(opt in statistics.columns for opt in scipy_columns)
    else:
        assert all(opt not in statistics.columns for opt in scipy_columns)
