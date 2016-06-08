# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from astropy.tests.helper import pytest
from astropy.units import Quantity
from astropy.wcs import WCS
from astropy.wcs import _wcs

from ..wcs import wcs_compare, pix2world


def test_wcs_compare():
    # These constants are also defined in the private module astropy.wcs._wcs
    # but I feel safer if these are defined here.
    WCSCOMPARE_ANCILLARY = 1  # bit repr: 001
    WCSCOMPARE_TILING = 2     # bit repr: 010
    WCSCOMPARE_CRPIX = 4      # bit repr: 100

    # Make sure these are identical to the astropy ones.
    assert _wcs.WCSCOMPARE_ANCILLARY == WCSCOMPARE_ANCILLARY
    assert _wcs.WCSCOMPARE_TILING == WCSCOMPARE_TILING
    assert _wcs.WCSCOMPARE_CRPIX == WCSCOMPARE_CRPIX

    # Now test the comparison function.
    wcs1 = WCS()
    wcs2 = WCS()
    assert wcs_compare(wcs1, wcs2)

    # Insert different observation dates.
    wcs1.wcs.dateobs = '100'
    wcs2.wcs.dateobs = '101'
    assert not wcs_compare(wcs1, wcs2)

    # Ignore the observation date:
    assert wcs_compare(wcs1, wcs2, cmp=WCSCOMPARE_ANCILLARY)

    # Ignore CRPIXja differences (but not the observation date)
    assert not wcs_compare(wcs1, wcs2, cmp=WCSCOMPARE_TILING)
    assert not wcs_compare(wcs1, wcs2, cmp=WCSCOMPARE_CRPIX)

    # Ignore CRPIXja differences and observation date
    assert wcs_compare(wcs1, wcs2, cmp=(WCSCOMPARE_TILING |
                                        WCSCOMPARE_ANCILLARY))
    assert wcs_compare(wcs1, wcs2, cmp=WCSCOMPARE_CRPIX | WCSCOMPARE_ANCILLARY)
    assert wcs_compare(wcs1, wcs2, cmp=(WCSCOMPARE_TILING | WCSCOMPARE_CRPIX |
                                        WCSCOMPARE_ANCILLARY))


def test_pix2world():
    wcs1D = WCS(naxis=1)
    wcs1D.wcs.crpix = [10]
    wcs1D.wcs.crval = [2]
    wcs1D.wcs.cunit = ['nm']
    wcs1D.wcs.cdelt = [1.5]

    data = np.ones(6)

    with pytest.raises(ValueError):
        pix2world(wcs1D, data, mode='fun')

    ref = np.array([-11.5, -10., -8.5, -7., -5.5, -4.])
    np.testing.assert_array_equal(ref, pix2world(wcs1D, data)[0].value)
    np.testing.assert_array_equal(ref, pix2world(wcs1D, data, 'wcs')[0].value)
    assert pix2world(wcs1D, data)[0].unit == 'nm'
