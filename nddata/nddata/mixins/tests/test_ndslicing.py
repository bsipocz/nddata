# Licensed under a 3-clause BSD style license - see LICENSE.rst

# TEST_UNICODE_LITERALS

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from numpy.testing import assert_array_equal

from astropy import units as u
from astropy.tests.helper import pytest
from astropy.wcs import WCS

from ... import NDData
from ...nduncertainty_stddev import StdDevUncertainty
from ...nduncertainty_unknown import UnknownUncertainty
from ...meta.nduncertainty_meta import NDUncertainty


# Just add the Mixin to NDData
# TODO: Make this use NDDataRef instead!
# class NDDataSliceable(NDSlicingMixin, NDData):
#     pass
NDDataSliceable = NDData


# Just some uncertainty (following the StdDevUncertainty implementation of
# storing the uncertainty in a propery 'array') with slicing.
class SomeUncertainty(NDUncertainty):

    @property
    def uncertainty_type(self):
        return 'fake'

    def _propagate_add(self, data, final_data):
        pass

    def _propagate_subtract(self, data, final_data):
        pass

    def _propagate_multiply(self, data, final_data):
        pass

    def _propagate_divide(self, data, final_data):
        pass


def test_slicing_only_data():
    data = np.arange(10)
    nd = NDDataSliceable(data)
    nd2 = nd[2:5]
    assert_array_equal(data[2:5], nd2.data)


def test_slicing_data_scalar_fail():
    data = np.array(10)
    nd = NDDataSliceable(data)
    with pytest.raises(TypeError):  # as exc
        nd[:]
    # assert exc.value.args[0] == 'Scalars cannot be sliced.'


def test_slicing_1ddata_ndslice():
    data = np.array([10, 20])
    nd = NDDataSliceable(data)
    # Standard numpy warning here:
    with pytest.raises(IndexError):
        nd[:, :]


@pytest.mark.parametrize('prop_name', ['mask', 'wcs', 'uncertainty'])
def test_slicing_1dmask_ndslice(prop_name):
    # Data is 2d but mask only 1d so this should let the IndexError when
    # slicing the mask rise to the user.
    data = np.ones((3, 3))
    kwarg = {prop_name: np.ones(3)}
    nd = NDDataSliceable(data, **kwarg)
    # Standard numpy warning here:
    with pytest.raises(IndexError):
        nd[:, :]


def test_slicing_all_npndarray_1d():
    data = np.arange(10)
    mask = data > 3
    uncertainty = np.linspace(10, 20, 10)
    wcs = np.linspace(1, 1000, 10)
    # Just to have them too
    unit = u.s
    meta = {'observer': 'Brian'}

    nd = NDDataSliceable(data, mask=mask, uncertainty=uncertainty, wcs=wcs,
                         unit=unit, meta=meta)
    nd2 = nd[2:5]
    assert_array_equal(data[2:5], nd2.data)
    assert_array_equal(mask[2:5], nd2.mask)
    assert_array_equal(uncertainty[2:5], nd2.uncertainty.data)
    assert_array_equal(wcs[2:5], nd2.wcs)
    assert unit is nd2.unit
    assert meta == nd.meta


def test_slicing_all_npndarray_nd():
    # See what happens for multidimensional properties
    data = np.arange(1000).reshape(10, 10, 10)
    mask = data > 3
    uncertainty = np.linspace(10, 20, 1000).reshape(10, 10, 10)
    wcs = np.linspace(1, 1000, 1000).reshape(10, 10, 10)
    flags = np.zeros(data.shape, dtype=bool)

    nd = NDDataSliceable(data, mask=mask, uncertainty=uncertainty, wcs=wcs,
                         flags=flags)
    # Slice only 1D
    nd2 = nd[2:5]
    assert_array_equal(data[2:5], nd2.data)
    assert_array_equal(mask[2:5], nd2.mask)
    assert_array_equal(uncertainty[2:5], nd2.uncertainty.data)
    assert_array_equal(wcs[2:5], nd2.wcs)
    assert_array_equal(flags[2:5], nd2.flags)
    # Slice 3D
    nd2 = nd[2:5, :, 4:7]
    assert_array_equal(data[2:5, :, 4:7], nd2.data)
    assert_array_equal(mask[2:5, :, 4:7], nd2.mask)
    assert_array_equal(uncertainty[2:5, :, 4:7], nd2.uncertainty.data)
    assert_array_equal(wcs[2:5, :, 4:7], nd2.wcs)
    assert_array_equal(flags[2:5, :, 4:7], nd2.flags)


def test_slicing_all_npndarray_shape_diff():
    data = np.arange(10)
    mask = (data > 3)[0:9]
    uncertainty = np.linspace(10, 20, 15)
    wcs = np.linspace(1, 1000, 12)
    flags = np.zeros(15)

    nd = NDDataSliceable(data, mask=mask, uncertainty=uncertainty, wcs=wcs,
                         flags=flags)
    nd2 = nd[2:5]
    assert_array_equal(data[2:5], nd2.data)
    # All are sliced even if the shapes differ (no Info)
    assert_array_equal(mask[2:5], nd2.mask)
    assert_array_equal(uncertainty[2:5], nd2.uncertainty.data)
    assert_array_equal(wcs[2:5], nd2.wcs)
    assert_array_equal(flags[2:5], nd2.flags)


def test_slicing_all_something_wrong():
    data = np.arange(10)
    mask = [False]*10
    uncertainty = {'rdnoise': 2.9, 'gain': 1.4}
    wcs = 145 * u.degree
    flags = None

    nd = NDDataSliceable(data, mask=mask, uncertainty=uncertainty, wcs=wcs,
                         flags=flags)
    nd2 = nd[2:5]
    # Sliced properties:
    assert_array_equal(data[2:5], nd2.data)
    assert_array_equal(mask[2:5], nd2.mask)
    # Not sliced attributes (they will raise a Info nevertheless)
    uncertainty is nd2.uncertainty
    assert_array_equal(wcs, nd2.wcs)
    assert flags is nd2.flags


def test_boolean_slicing():
    data = np.arange(10)
    mask = data.copy()
    uncertainty = StdDevUncertainty(data.copy())
    wcs = data.copy()
    flags = data.astype(bool)
    nd = NDDataSliceable(data, mask=mask, uncertainty=uncertainty, wcs=wcs,
                         flags=flags)

    nd2 = nd[(nd.data >= 3) & (nd.data < 8)]
    assert_array_equal(data[3:8], nd2.data)
    assert_array_equal(mask[3:8], nd2.mask)
    assert_array_equal(wcs[3:8], nd2.wcs)
    assert_array_equal(uncertainty.data[3:8], nd2.uncertainty.data)
    assert_array_equal(flags[3:8], nd2.flags)


def test_slice_cutout_1d():
    wcs1D = WCS(naxis=1)
    wcs1D.wcs.crpix = [1]
    wcs1D.wcs.crval = [1]
    wcs1D.wcs.cunit = ["deg"]
    wcs1D.wcs.cdelt = [1.5]

    data = np.arange(100)

    ndd = NDData(data, wcs=wcs1D)

    # Fails with multid position or shape
    with pytest.raises(ValueError):
        ndd.slice_cutout([1, 2], 1)
    with pytest.raises(ValueError):
        ndd.slice_cutout(1, [1, 2])

    # Test all 4 cases: unitless position/shape and quantity position/shape
    ndd_cutout = ndd.slice_cutout(10, 10)
    assert_array_equal(ndd_cutout.data, np.arange(10, 20))
    assert ndd_cutout.wcs.wcs.crpix[0] == -9

    ndd_cutout = ndd.slice_cutout(10, 13.5*u.deg)
    assert_array_equal(ndd_cutout.data, np.arange(10, 20))
    assert ndd_cutout.wcs.wcs.crpix[0] == -9

    ndd_cutout = ndd.slice_cutout(16.5*u.deg, 10)
    assert_array_equal(ndd_cutout.data, np.arange(10, 20))
    assert ndd_cutout.wcs.wcs.crpix[0] == -9

    ndd_cutout = ndd.slice_cutout(16.5*u.deg, 13.5*u.deg)
    assert_array_equal(ndd_cutout.data, np.arange(10, 20))
    assert ndd_cutout.wcs.wcs.crpix[0] == -9

    # Test 1-element tuples and lists work exactly like giving scalars in the
    # 1D case:
    ndd_cutout_ref = ndd.slice_cutout(10, 10)
    ndd_cutout = ndd.slice_cutout((10, ), 10)
    assert_array_equal(ndd_cutout.data, ndd_cutout_ref.data)
    assert ndd_cutout.wcs.wcs.crpix[0] == ndd_cutout_ref.wcs.wcs.crpix[0]

    ndd_cutout = ndd.slice_cutout(10, (10, ))
    assert_array_equal(ndd_cutout.data, ndd_cutout_ref.data)
    assert ndd_cutout.wcs.wcs.crpix[0] == ndd_cutout_ref.wcs.wcs.crpix[0]

    ndd_cutout = ndd.slice_cutout((10, ), (10, ))
    assert_array_equal(ndd_cutout.data, ndd_cutout_ref.data)
    assert ndd_cutout.wcs.wcs.crpix[0] == ndd_cutout_ref.wcs.wcs.crpix[0]

    ndd_cutout = ndd.slice_cutout([10], [10])
    assert_array_equal(ndd_cutout.data, ndd_cutout_ref.data)
    assert ndd_cutout.wcs.wcs.crpix[0] == ndd_cutout_ref.wcs.wcs.crpix[0]


def test_slice_cutout_2d():
    wcs2D = WCS(naxis=2)
    wcs2D.wcs.crpix = [1, 1]
    wcs2D.wcs.crval = [1, 1]
    wcs2D.wcs.cunit = ["deg", "deg"]
    wcs2D.wcs.cdelt = [1.5, 1.5]

    data = np.arange(10000).reshape(100, 100)

    ndd = NDData(data, wcs=wcs2D)

    # Fails with scalar
    with pytest.raises(TypeError):
        ndd.slice_cutout([1, 2], 1)
    with pytest.raises(TypeError):
        ndd.slice_cutout(1, [1, 2])

    # Fails with wrong dimensions
    with pytest.raises(ValueError):
        ndd.slice_cutout([1, 2], [1])
    with pytest.raises(ValueError):
        ndd.slice_cutout([1], [1, 2])

    # Test all 4 cases: unitless position/shape and quantity position/shape
    ndd_cutout = ndd.slice_cutout((10, 10), (10, 10))
    assert_array_equal(ndd_cutout.data, data[10:20, 10:20])
    assert ndd_cutout.wcs.wcs.crpix[0] == -9
    assert ndd_cutout.wcs.wcs.crpix[1] == -9

    ndd_cutout = ndd.slice_cutout((10, 10), (13.5*u.deg, 13.5*u.deg))
    assert_array_equal(ndd_cutout.data, data[10:20, 10:20])
    assert ndd_cutout.wcs.wcs.crpix[0] == -9
    assert ndd_cutout.wcs.wcs.crpix[1] == -9

    ndd_cutout = ndd.slice_cutout((16.5*u.deg, 16.5*u.deg), (10, 10))
    assert_array_equal(ndd_cutout.data, data[10:20, 10:20])
    assert ndd_cutout.wcs.wcs.crpix[0] == -9
    assert ndd_cutout.wcs.wcs.crpix[1] == -9

    ndd_cutout = ndd.slice_cutout((16.5*u.deg, 16.5*u.deg),
                                  (13.5*u.deg, 13.5*u.deg))
    assert_array_equal(ndd_cutout.data, data[10:20, 10:20])
    assert ndd_cutout.wcs.wcs.crpix[0] == -9
    assert ndd_cutout.wcs.wcs.crpix[1] == -9


def test_slice_cutout_3d():
    wcs3D = WCS(naxis=3)
    wcs3D.wcs.crpix = [1, 1, 1]
    wcs3D.wcs.crval = [1, 1, 1]
    wcs3D.wcs.cunit = ["deg", "deg", "deg"]
    wcs3D.wcs.cdelt = [1.5, 1.5, 1.5]

    data = np.arange(1000000).reshape(100, 100, 100)

    ndd = NDData(data, wcs=wcs3D)

    # Fails with scalar
    with pytest.raises(TypeError):
        ndd.slice_cutout([1, 2], 1)
    with pytest.raises(TypeError):
        ndd.slice_cutout(1, [1, 2])

    # Fails with wrong dimensions
    with pytest.raises(ValueError):
        ndd.slice_cutout([1, 2, 3], [1])
    with pytest.raises(ValueError):
        ndd.slice_cutout([1], [1, 2, 3])

    # Test all 4 cases: unitless position/shape and quantity position/shape
    ndd_cutout = ndd.slice_cutout((10, 10, 10), (10, 10, 10))
    assert_array_equal(ndd_cutout.data, data[10:20, 10:20, 10:20])
    assert ndd_cutout.wcs.wcs.crpix[0] == -9
    assert ndd_cutout.wcs.wcs.crpix[1] == -9
    assert ndd_cutout.wcs.wcs.crpix[2] == -9

    ndd_cutout = ndd.slice_cutout((10, 10, 10),
                                  (13.5*u.deg, 13.5*u.deg, 13.5*u.deg))
    assert_array_equal(ndd_cutout.data, data[10:20, 10:20, 10:20])
    assert ndd_cutout.wcs.wcs.crpix[0] == -9
    assert ndd_cutout.wcs.wcs.crpix[1] == -9
    assert ndd_cutout.wcs.wcs.crpix[2] == -9

    ndd_cutout = ndd.slice_cutout((16.5*u.deg, 16.5*u.deg, 16.5*u.deg),
                                  (10, 10, 10))
    assert_array_equal(ndd_cutout.data, data[10:20, 10:20, 10:20])
    assert ndd_cutout.wcs.wcs.crpix[0] == -9
    assert ndd_cutout.wcs.wcs.crpix[1] == -9
    assert ndd_cutout.wcs.wcs.crpix[2] == -9

    ndd_cutout = ndd.slice_cutout((16.5*u.deg, 16.5*u.deg, 16.5*u.deg),
                                  (13.5*u.deg, 13.5*u.deg, 13.5*u.deg))
    assert_array_equal(ndd_cutout.data, data[10:20, 10:20, 10:20])
    assert ndd_cutout.wcs.wcs.crpix[0] == -9
    assert ndd_cutout.wcs.wcs.crpix[1] == -9
    assert ndd_cutout.wcs.wcs.crpix[2] == -9
