# Licensed under a 3-clause BSD style license - see LICENSE.rst

# TEST_UNICODE_LITERALS

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from ... import NDData, StdDevUncertainty
from astropy.tests.helper import pytest
from astropy import units as u


def test_fail():
    # If the instance has no unit one cannot convert it
    ndd = NDData(100)
    with pytest.raises(TypeError):
        ndd.convert_unit_to('m')


def test_data():
    # Data with units are convertible with astropy.units framework, just do
    # simple tests. The heavy testing is done within astropy
    ndd = NDData(100, unit='cm')
    ndd2 = ndd.convert_unit_to('m')
    assert ndd2.data == 1
    assert ndd2.unit == u.m


def test_data_with_attributes():
    data = np.array([-2, 0, 5.5])
    mask = np.array([True, False, True], dtype=bool)
    unce = StdDevUncertainty([1, 1, 1])
    meta = {'test': 'test'}
    unit = u.m
    wcs = 10
    flags = 10
    ndd = NDData(data, unit=unit, mask=mask, meta=meta, uncertainty=unce,
                 wcs=wcs, flags=flags)

    ndd2 = ndd.convert_unit_to(u.mm)
    np.testing.assert_array_equal(ndd.data * 1000, ndd2.data)
    np.testing.assert_array_equal(ndd.mask, ndd2.mask)
    # TODO: This should be converted too but no support for this yet...
    np.testing.assert_array_equal(ndd.uncertainty.data, ndd2.uncertainty.data)
    assert ndd.meta == ndd2.meta
    assert ndd.wcs == ndd2.wcs
    assert ndd.flags == ndd2.flags
    assert ndd2.unit == u.mm
