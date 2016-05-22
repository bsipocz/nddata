# Licensed under a 3-clause BSD style license - see LICENSE.rst

# TEST_UNICODE_LITERALS

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from ... import (NDData, StdDevUncertainty, VarianceUncertainty,
                 UnknownUncertainty, RelativeUncertainty)
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
    assert ndd.is_identical(ndd2, strict=False)


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
    np.testing.assert_array_equal(ndd.uncertainty.data * 1000,
                                  ndd2.uncertainty.data)
    assert ndd.meta == ndd2.meta
    assert ndd.wcs == ndd2.wcs
    assert ndd.flags == ndd2.flags
    assert ndd2.unit == u.mm
    assert ndd.is_identical(ndd2, strict=False)


def test_copies():
    # Check that it copies the data with the special case that the unit is
    # identical to the original one.
    ndd = NDData(np.arange(5), unit='m')
    ndd2 = ndd.convert_unit_to('m')
    ndd3 = ndd.convert_unit_to(u.m)
    assert ndd.is_identical(ndd2, strict=False)
    assert ndd.is_identical(ndd3, strict=False)

    ndd.data[0] = 10
    assert ndd2.data[0] == 0
    assert ndd3.data[0] == 0


def test_uncertainties_rel():
    # relative uncertainties are not allowed to have a unit, so extensive
    # testing makes no sense. Make sure it's copied though.
    ndd = NDData([100], unit='m',
                 uncertainty=RelativeUncertainty([0.2], copy=True))
    ndd2 = ndd.convert_unit_to('cm')
    np.testing.assert_array_equal(ndd.uncertainty.data,
                                  ndd2.uncertainty.data)
    assert ndd.is_identical(ndd2, strict=False)
    # make sure it's a copy!
    ndd.uncertainty.data[0] = 50
    assert ndd2.uncertainty.data[0] == 0.2


def test_uncertainties_var():
    # uncertainty has a different unit than the data and is also the target.
    ndd = NDData([100], unit='m',
                 uncertainty=VarianceUncertainty([100], unit='cm*cm',
                                                 copy=True))
    ndd2 = ndd.convert_unit_to('cm')
    np.testing.assert_array_equal(ndd.uncertainty.data,
                                  ndd2.uncertainty.data)
    assert ndd2.uncertainty.effective_unit == u.cm * u.cm
    assert ndd.is_identical(ndd2, strict=False)
    # make sure it's a copy!
    ndd.uncertainty.data[0] = 50
    assert ndd2.uncertainty.data[0] == 100

    # same unit (explicit)
    ndd = NDData([100], unit='m',
                 uncertainty=VarianceUncertainty([100], unit='m*m', copy=True))
    ndd2 = ndd.convert_unit_to('cm')
    np.testing.assert_array_equal(ndd.uncertainty.data * 100 * 100,
                                  ndd2.uncertainty.data)
    assert ndd2.uncertainty.effective_unit == u.cm * u.cm
    assert ndd.is_identical(ndd2, strict=False)

    # same unit (implicit)
    ndd = NDData([100], unit='m',
                 uncertainty=VarianceUncertainty([100], copy=True))
    ndd2 = ndd.convert_unit_to('cm')
    np.testing.assert_array_equal(ndd.uncertainty.data * 100 * 100,
                                  ndd2.uncertainty.data)
    assert ndd2.uncertainty.effective_unit == u.cm * u.cm
    assert ndd.is_identical(ndd2, strict=False)


def test_uncertainties_stddev():
    # uncertainty has a different unit than the data and is also the target.
    ndd = NDData([100], unit='m',
                 uncertainty=StdDevUncertainty([100], unit='cm', copy=True))
    ndd2 = ndd.convert_unit_to('cm')
    np.testing.assert_array_equal(ndd.uncertainty.data,
                                  ndd2.uncertainty.data)
    assert ndd2.uncertainty.effective_unit == u.cm
    assert ndd.is_identical(ndd2, strict=False)
    # make sure it's a copy!
    ndd.uncertainty.data[0] = 50
    assert ndd2.uncertainty.data[0] == 100

    # same unit (explicit)
    ndd = NDData([100], unit='m',
                 uncertainty=StdDevUncertainty([100], unit='m', copy=True))
    ndd2 = ndd.convert_unit_to('cm')
    np.testing.assert_array_equal(ndd.uncertainty.data * 100,
                                  ndd2.uncertainty.data)
    assert ndd2.uncertainty.effective_unit == u.cm
    assert ndd.is_identical(ndd2, strict=False)

    # same unit (implicit)
    ndd = NDData([100], unit='m',
                 uncertainty=StdDevUncertainty([100], copy=True))
    ndd2 = ndd.convert_unit_to('cm')
    np.testing.assert_array_equal(ndd.uncertainty.data * 100,
                                  ndd2.uncertainty.data)
    assert ndd2.uncertainty.effective_unit == u.cm
    assert ndd.is_identical(ndd2, strict=False)


def test_uncertainties_unknown():
    uncertainty = np.array([100])
    # uncertainty has a different unit than the data and is also the target.
    ndd = NDData([100], unit='m',
                 uncertainty=UnknownUncertainty(uncertainty, unit='cm',
                                                copy=True))
    ndd2 = ndd.convert_unit_to('cm')
    np.testing.assert_array_equal(ndd.uncertainty.data,
                                  ndd2.uncertainty.data)
    assert ndd.is_identical(ndd2, strict=False)
    # make sure it's a copy!
    ndd.uncertainty.data[0] = 50
    assert ndd2.uncertainty.data[0] == 100

    # same unit (explicit)
    ndd = NDData([100], unit='m',
                 uncertainty=UnknownUncertainty(uncertainty, unit='m',
                                                copy=True))
    ndd2 = ndd.convert_unit_to('cm')
    np.testing.assert_array_equal(ndd.uncertainty.data * 100,
                                  ndd2.uncertainty.data)
    assert ndd.is_identical(ndd2, strict=False)

    # THIS case differs from stddev because we don't know what uncertainty
    # is set, so we don't know how the unit relates to the parent, we expect
    # a copy without alteration.
    ndd = NDData([100], unit='m',
                 uncertainty=UnknownUncertainty(uncertainty, copy=True))
    ndd2 = ndd.convert_unit_to('cm')
    np.testing.assert_array_equal(ndd.uncertainty.data,
                                  ndd2.uncertainty.data)
    assert ndd.is_identical(ndd2, strict=False)
    # make sure it's a copy!
    ndd.uncertainty.data[0] = 50
    assert ndd2.uncertainty.data[0] == 100
