# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from astropy.tests.helper import pytest
from astropy.utils.misc import NumpyRNGContext

from ... import OPT_DEPS

from ..numbautils import stats_one_pass as numba_stats


@pytest.mark.xfail(OPT_DEPS['NUMBA'], strict=True,
                   reason="this is what happens if no numba is installed.")
def test_stats_fail_without_numba():
    data = np.ones((3, 3))
    # no statistics without numba
    with pytest.raises(ImportError):
        numba_stats(data)


def test_stats_fail():
    data = np.array(1)
    # There can be no statistic with one element
    with pytest.raises(TypeError):
        numba_stats(data)


@pytest.mark.xfail(not OPT_DEPS['NUMBA'], strict=True, reason="numba required")
def test_stats_numba():
    with NumpyRNGContext(12345):
        data = np.random.random(40)

    assert numba_stats(data)[0] == data.size
    np.testing.assert_allclose(numba_stats(data)[1], np.sum(data))
    np.testing.assert_allclose(numba_stats(data)[2], np.mean(data))
    assert numba_stats(data)[3] == np.min(data)
    assert numba_stats(data)[4] == np.max(data)
    np.testing.assert_allclose(numba_stats(data)[5], np.std(data))
    np.testing.assert_allclose(numba_stats(data)[6], np.var(data))


@pytest.mark.xfail(not OPT_DEPS['NUMBA'], strict=True, reason="numba required")
def test_stats_numba_nd():
    with NumpyRNGContext(12345):
        data = np.random.random((10, 10))  # will be ravelled.

    assert numba_stats(data)[0] == data.size
    np.testing.assert_allclose(numba_stats(data)[1], np.sum(data))
    np.testing.assert_allclose(numba_stats(data)[2], np.mean(data))
    assert numba_stats(data)[3] == np.min(data)
    assert numba_stats(data)[4] == np.max(data)
    np.testing.assert_allclose(numba_stats(data)[5], np.std(data))
    np.testing.assert_allclose(numba_stats(data)[6], np.var(data))


@pytest.mark.xfail(not OPT_DEPS['NUMBA'], strict=True, reason="numba required")
def test_stats_numba_list():
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # normal python list

    assert numba_stats(data)[0] == len(data)
    np.testing.assert_allclose(numba_stats(data)[1], np.sum(data))
    np.testing.assert_allclose(numba_stats(data)[2], np.mean(data))
    assert numba_stats(data)[3] == np.min(data)
    assert numba_stats(data)[4] == np.max(data)
    np.testing.assert_allclose(numba_stats(data)[5], np.std(data))
    np.testing.assert_allclose(numba_stats(data)[6], np.var(data))


@pytest.mark.xfail(not OPT_DEPS['NUMBA'], strict=True, reason="numba required")
def test_stats_numba_cancellation():
    with NumpyRNGContext(12345):
        # mean very high compared to variance because I added 1e9
        data = np.random.random((1000000)) + 1e9

    assert numba_stats(data)[0] == len(data)
    np.testing.assert_allclose(numba_stats(data)[1], np.sum(data))
    np.testing.assert_allclose(numba_stats(data)[2], np.mean(data))
    assert numba_stats(data)[3] == np.min(data)
    assert numba_stats(data)[4] == np.max(data)
    np.testing.assert_allclose(numba_stats(data)[5], np.std(data))
    np.testing.assert_allclose(numba_stats(data)[6], np.var(data))
