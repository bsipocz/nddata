# Licensed under a 3-clause BSD style license - see LICENSE.rst

# TEST_UNICODE_LITERALS

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from copy import deepcopy
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from ...nduncertainty_stddev import StdDevUncertainty
from ...nduncertainty_unknown import UnknownUncertainty
from ...nduncertainty_var import VarianceUncertainty
from ...nduncertainty_relstd import RelativeUncertainty
from ...exceptions import IncompatibleUncertaintiesException
from ...meta.nduncertainty_meta import NDUncertaintyPropagatable, NDUncertainty
from ...contexts import ContextArithmeticDefaults

from ... import NDDataBase
from .. import NDArithmeticPyOpsMixin
from astropy.units import UnitsError, Quantity
from astropy.tests.helper import pytest
from astropy import units as u
from astropy.wcs import WCS

import astropy
from distutils.version import LooseVersion
numpy_1_10 = LooseVersion(np.__version__) >= LooseVersion('1.10')


class NDDataArithmetic(NDArithmeticPyOpsMixin, NDDataBase):
    pass


def create_ndd(all_attrs=True):
    data = np.array([-2, 0, 5.5])
    if all_attrs:
        mask = np.array([True, False, True], dtype=bool)
        meta = {'test': 'test'}
        unce = StdDevUncertainty([1, 0.4, 1.1])
        unit = u.dimensionless_unscaled
        wcs = None
        flags = None
        return NDDataArithmetic(data, unit=unit, mask=mask, meta=meta,
                                uncertainty=unce, wcs=wcs, flags=flags,
                                copy=False)
    return NDDataArithmetic(data, copy=False)


def compare_ndd_identical(ndd1, ndd2):
    assert isinstance(ndd1, NDDataBase)
    assert isinstance(ndd2, NDDataBase)

    np.testing.assert_array_equal(ndd1.data, ndd2.data)

    if isinstance(ndd1.mask, np.ndarray):
        np.testing.assert_array_equal(ndd1.mask, ndd2.mask)
    else:
        assert ndd1.mask == ndd2.mask

    assert ndd1.unit == ndd2.unit

    assert ndd1.meta.__class__ is ndd2.meta.__class__
    assert ndd1.meta == ndd2.meta

    if isinstance(ndd1.wcs, np.ndarray):
        np.testing.assert_array_equal(ndd1.wcs, ndd2.wcs)
    elif isinstance(ndd1.wcs, WCS):
        assert ndd1.wcs.wcs.compare(ndd2.wcs.wcs)
    else:
        assert ndd1.wcs == ndd2.wcs

    if isinstance(ndd1.flags, np.ndarray):
        np.testing.assert_array_equal(ndd1.flags, ndd2.flags)
    else:
        assert ndd1.flags == ndd2.flags

    if isinstance(ndd1.uncertainty, NDUncertaintyPropagatable):
        np.testing.assert_array_equal(ndd1.uncertainty.data,
                                      ndd2.uncertainty.data)
        assert ndd1.uncertainty.unit == ndd2.uncertainty.unit
    elif isinstance(ndd1.uncertainty, NDUncertainty):
        if isinstance(ndd1.uncertainty.data, np.ndarray):
            np.testing.assert_array_equal(ndd1.uncertainty.data,
                                          ndd2.uncertainty.data)
        else:
            assert ndd1.uncertainty.data == ndd2.uncertainty.data
        assert ndd1.uncertainty.unit == ndd2.uncertainty.unit
    else:
        assert ndd1.uncertainty == ndd2.uncertainty


@pytest.mark.parametrize(('op2'), [1, 1.5, np.array(2), np.array([3]),
                                   np.ma.array(2, mask=False),
                                   5 * u.dimensionless_unscaled])
def test_arithmetic_ops(op2):
    if not numpy_1_10 and isinstance(op2, np.ma.MaskedArray):
        pytest.xfail("masked arrays didn't respect numpy priority yet...")

    ndd = create_ndd()
    compare_ndd_identical(ndd.add(op2),      ndd + op2)
    compare_ndd_identical(ndd.subtract(op2), ndd - op2)
    compare_ndd_identical(ndd.multiply(op2), ndd * op2)
    compare_ndd_identical(ndd.divide(op2),   ndd / op2)
    compare_ndd_identical(ndd.power(op2),    ndd ** op2)

    # and reverse
    compare_ndd_identical(ndd.add(op2, ndd),      op2 + ndd)
    compare_ndd_identical(ndd.subtract(op2, ndd), op2 - ndd)
    compare_ndd_identical(ndd.multiply(op2, ndd), op2 * ndd)
    compare_ndd_identical(ndd.divide(op2, ndd),   op2 / ndd)
    compare_ndd_identical(ndd.power(op2, ndd),    op2 ** ndd)


@pytest.mark.parametrize(('op2'), [1, 1.5, np.array(2), np.array([3]),
                                   np.ma.array(2, mask=False),
                                   5 * u.dimensionless_unscaled])
def test_arithmetic_ops_optional_kwargs(op2):
    if not numpy_1_10 and isinstance(op2, np.ma.MaskedArray):
        pytest.xfail("masked arrays didn't respect numpy priority yet...")

    ndd = create_ndd()

    # Let us test the context manager ... to change the defaults.
    with ContextArithmeticDefaults() as d:
        d['handle_mask'] = None
        d['propagate_uncertainties'] = None

        opkwargs = {'handle_mask': None, 'propagate_uncertainties': None}
        compare_ndd_identical(ndd.add(op2, **opkwargs),      ndd + op2)
        compare_ndd_identical(ndd.subtract(op2, **opkwargs), ndd - op2)
        compare_ndd_identical(ndd.multiply(op2, **opkwargs), ndd * op2)
        compare_ndd_identical(ndd.divide(op2, **opkwargs),   ndd / op2)
        compare_ndd_identical(ndd.power(op2, **opkwargs),    ndd ** op2)

        # and reverse
        compare_ndd_identical(ndd.add(op2, ndd, **opkwargs),      op2 + ndd)
        compare_ndd_identical(ndd.subtract(op2, ndd, **opkwargs), op2 - ndd)
        compare_ndd_identical(ndd.multiply(op2, ndd, **opkwargs), op2 * ndd)
        compare_ndd_identical(ndd.divide(op2, ndd, **opkwargs),   op2 / ndd)
        compare_ndd_identical(ndd.power(op2, ndd, **opkwargs),    op2 ** ndd)


def test_arithmetic_boring_ops():
    ndd = create_ndd()
    ndd2 = ndd.copy()
    # There is also negate and pos ...
    compare_ndd_identical(ndd, +ndd)

    ndd2.data = ndd.data * -1
    compare_ndd_identical(ndd2, -ndd)

    ndd2.data = np.abs(ndd2.data)
    compare_ndd_identical(ndd2, abs(ndd))
