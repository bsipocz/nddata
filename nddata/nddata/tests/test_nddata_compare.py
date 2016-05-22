# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from collections import OrderedDict

import astropy.units as u

from ..nddata_base import NDDataBase
from ..nddata import NDData
from ..nduncertainty_stddev import StdDevUncertainty


def create_wcs(disturb=False):
    """Taken from the astropy documentation:
    http://docs.astropy.org/en/stable/wcs/

    With a slightly modification to disturb the values a bit. :-)
    """
    from astropy import wcs
    # Create a new WCS object.  The number of axes must be set
    # from the start
    w = wcs.WCS(naxis=2)

    # Set up an "Airy's zenithal" projection
    # Vector properties may be set with Python lists, or Numpy arrays
    if disturb:
        w.wcs.crpix = [-234.75, 5.3393]
    else:
        w.wcs.crpix = [-234.75, 8.3393]
    w.wcs.cdelt = np.array([-0.066667, 0.066667])
    w.wcs.crval = [0, -90]
    w.wcs.ctype = ["RA---AIR", "DEC--AIR"]
    w.wcs.set_pv([(2, 1, 45.0)])
    return w


def test_identical_after_copy():
    ndd = NDDataBase(10)
    ndd2 = ndd.copy()
    assert ndd.is_identical(ndd2)

    ndd = NDDataBase([10, 20, 30])
    ndd2 = ndd.copy()
    assert ndd.is_identical(ndd2)


def test_identical_identity():
    ndd = NDDataBase(10)
    ndd2 = ndd
    assert ndd.is_identical(ndd2)


def test_not_identical_class():
    ndd = NDDataBase(10)
    ndd2 = NDData(ndd)
    assert not ndd.is_identical(ndd2)
    assert ndd.is_identical(ndd2, strict=False)


def test_not_identical_data():
    # Case 1: One data is None, the other isn't
    ndd1 = NDDataBase(10)
    ndd2 = NDDataBase(None)
    assert not ndd1.is_identical(ndd2)

    ndd1 = NDDataBase(None)
    ndd2 = NDDataBase(10)
    assert not ndd1.is_identical(ndd2)

    # Case 2: Both are numpy ndarrays but shape differs
    ndd1 = NDDataBase([10])
    ndd2 = NDDataBase([10, 11])
    assert not ndd1.is_identical(ndd2)

    # Case 3: One value differs
    ndd1 = NDDataBase([10])
    ndd2 = NDDataBase([11])
    assert not ndd1.is_identical(ndd2)


def test_not_identical_unit():
    # Case 3: One value differs
    ndd1 = NDDataBase([100], unit='cm')
    ndd2 = NDDataBase([1], unit='m')
    assert not ndd1.is_identical(ndd2)
    assert ndd1.is_identical(ndd2, strict=False)

    ndd1 = NDDataBase([1])
    ndd2 = NDDataBase([1], unit=u.dimensionless_unscaled)
    assert not ndd1.is_identical(ndd2)
    assert ndd1.is_identical(ndd2, strict=False)

    ndd1 = NDDataBase([1], unit='')
    ndd2 = NDDataBase([1])
    assert not ndd1.is_identical(ndd2)
    assert ndd1.is_identical(ndd2, strict=False)


def test_not_identical_mask():
    # Case 1: One mask is None, the other isn't
    ndd1 = NDDataBase(1, mask=1)
    ndd2 = NDDataBase(1)
    assert not ndd1.is_identical(ndd2)

    # Case 2: both masks are np.ndarray but differ in shape
    ndd1 = NDDataBase(1, mask=np.array([True, False]))
    ndd2 = NDDataBase(1, mask=np.array([True]))
    assert not ndd1.is_identical(ndd2)

    # Case 2: both masks are np.ndarray but contain different values
    ndd1 = NDDataBase(1, mask=np.array([True, False]))
    ndd2 = NDDataBase(1, mask=np.array([True, True]))
    assert not ndd1.is_identical(ndd2)


def test_not_identical_flags():
    # Case 1: One flags is None, the other isn't
    ndd1 = NDDataBase(1, flags=1)
    ndd2 = NDDataBase(1)
    assert not ndd1.is_identical(ndd2)

    # Case 2: both flags are np.ndarray but differ in shape
    ndd1 = NDDataBase(1, flags=np.array([102, 15]))
    ndd2 = NDDataBase(1, flags=np.array([102]))
    assert not ndd1.is_identical(ndd2)

    # Case 2: both flags are np.ndarray but contain different values
    ndd1 = NDDataBase(1, flags=np.array([102, 14]))
    ndd2 = NDDataBase(1, flags=np.array([102, 15]))
    assert not ndd1.is_identical(ndd2)


def test_not_identical_wcs():
    # Case 1: One wcs is None, the other isn't
    ndd1 = NDDataBase(1, wcs=1)
    ndd2 = NDDataBase(1)
    assert not ndd1.is_identical(ndd2)

    # Case 2: both wcs are np.ndarray but differ in shape
    ndd1 = NDDataBase(1, wcs=np.array([1, 2]))
    ndd2 = NDDataBase(1, wcs=np.array([1]))
    assert not ndd1.is_identical(ndd2)

    # Case 2: both wcs are np.ndarray but contain different values
    ndd1 = NDDataBase(1, wcs=np.array([1, 2]))
    ndd2 = NDDataBase(1, wcs=np.array([1, 1]))
    assert not ndd1.is_identical(ndd2)

    # Make sure it passes with two identical wcs
    wcs1 = create_wcs()
    wcs2 = create_wcs()
    ndd1 = NDDataBase(1, wcs=wcs1)
    ndd2 = NDDataBase(1, wcs=wcs2)
    assert ndd1.is_identical(ndd2)

    # Now crash it
    wcs1 = create_wcs()
    wcs2 = create_wcs(disturb=True)
    ndd1 = NDDataBase(1, wcs=wcs1)
    ndd2 = NDDataBase(1, wcs=wcs2)
    assert not ndd1.is_identical(ndd2)


def test_not_identical_meta():
    # Case 1: Both meta have the same content but different classes
    meta = {'a': 10}

    ndd1 = NDDataBase(1, meta=meta)
    ndd2 = NDDataBase(1, meta=OrderedDict(meta))
    assert not ndd1.is_identical(ndd2)
    assert ndd1.is_identical(ndd2, strict=False)

    # Case 2: Both have the same class but different content
    ndd1 = NDDataBase(1, meta={'a': 1})
    ndd2 = NDDataBase(1, meta={'a': 2})
    assert not ndd1.is_identical(ndd2)


def test_not_identical_uncertainty():
    # Case 1: Test if they are the same :-)
    ndd1 = NDDataBase(1, uncertainty=10)
    ndd2 = NDDataBase(1, uncertainty=10)
    assert ndd1.is_identical(ndd2)

    # Test directly two uncertainties for equality - identity test doesn't work
    # when setting it on NDData because we explicitly made sure that it creates
    # a new class when the uncertainty already has a parent.
    uncert1 = StdDevUncertainty(10)
    uncert2 = uncert1
    assert uncert1.is_identical(uncert2)

    # Case 2: both uncertainties are np.ndarray but differ in shape
    ndd1 = NDDataBase(1, uncertainty=np.array([102, 15]))
    ndd2 = NDDataBase(1, uncertainty=np.array([102]))
    assert not ndd1.is_identical(ndd2)

    # Case 3: both uncertainties are np.ndarray but contain different values
    ndd1 = NDDataBase(1, uncertainty=np.array([102, 14]))
    ndd2 = NDDataBase(1, uncertainty=np.array([102, 15]))
    assert not ndd1.is_identical(ndd2)

    # Case 4: we need to provoke an attributeError, so we need unknown
    # uncertainties where the first one is a np.ndarray but the second one
    # isn't
    ndd1 = NDDataBase(1, uncertainty=np.array([102, 14]))
    ndd2 = NDDataBase(1, uncertainty=2)
    assert not ndd1.is_identical(ndd2)

    # and both have the same values but different classes
    ndd1 = NDDataBase(1, uncertainty=np.array(10))
    ndd2 = NDDataBase(1, uncertainty=StdDevUncertainty(10))
    assert not ndd1.is_identical(ndd2)
