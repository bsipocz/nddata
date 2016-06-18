# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import textwrap
from collections import OrderedDict, namedtuple

import numpy as np
from numpy.testing import assert_array_equal

from ..nddata_base import NDDataBase
from ..nduncertainty_stddev import StdDevUncertainty
from ..nduncertainty_unknown import UnknownUncertainty
from ..meta.nduncertainty_meta import NDUncertainty
from astropy.tests.helper import pytest
from astropy import units as u
from astropy.utils import NumpyRNGContext


class FakeUncertainty(NDUncertainty):

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


class FakeNumpyArray(object):
    """
    Anything that provides an numpy.ndarray-like interface should be savable as
    data.
    """
    def __array__(self):
        return np.array([1, 2, 3])


class MinimalUncertainty(object):
    """
    Define the minimum attributes acceptable as an uncertainty object.
    """
    def __init__(self, value):
        self._uncertainty = value

    @property
    def uncertainty_type(self):
        return "totally and completely fake"


class BadNDDataSubclass(NDDataBase):

    def __init__(self, data, uncertainty=None, mask=None, wcs=None,
                 meta=None, unit=None):
        self._data = data
        self._uncertainty = uncertainty
        self._mask = mask
        self._wcs = wcs
        self._unit = unit
        self._meta = meta


class NDDataInterface(object):

    def __init__(self, data, uncertainty=None, mask=None, wcs=None,
                 meta=None, unit=None, flags=None):
        self._data = data
        self._uncertainty = uncertainty
        self._mask = mask
        self._wcs = wcs
        self._unit = unit
        self._meta = meta
        self._flags = flags

    def __astropy_nddata__(self):
        return {'data': self._data, 'uncertainty': self._uncertainty,
                'mask': self._mask, 'unit': self._unit, 'wcs': self._wcs,
                'meta': self._meta, 'flags': self._flags}


class NDDataPartialInterface(NDDataInterface):

    def __astropy_nddata__(self):
        return {'data': self._data, 'unit': self._unit, 'meta': self._meta}


class NDDataBrokenInterface(NDDataInterface):
    # Broken means it has no data. Which is allowed but generally not
    # recommended.

    def __astropy_nddata__(self):
        return {'meta': self._meta, 'uncertainty': self._uncertainty,
                'mask': self._mask, 'unit': self._unit, 'wcs': self._wcs}


# Setter tests
def test_uncertainty_setter():
    nd = NDDataBase([1, 2, 3])
    good_uncertainty = MinimalUncertainty(5)
    nd.uncertainty = good_uncertainty
    assert nd.uncertainty is good_uncertainty
    # Check the fake uncertainty (minimal does not work since it has no
    # parent_nddata attribute from NDUncertainty)
    nd.uncertainty = FakeUncertainty(5)
    assert nd.uncertainty.parent_nddata is nd
    # Check that it works if the uncertainty was set during init
    nd = NDDataBase(nd)
    assert isinstance(nd.uncertainty, FakeUncertainty)
    nd.uncertainty = 10
    assert not isinstance(nd.uncertainty, FakeUncertainty)
    assert nd.uncertainty.data == 10


def test_mask_setter():
    # Since it just changes the _mask attribute everything should work
    nd = NDDataBase([1, 2, 3])
    nd.mask = True
    assert nd.mask
    nd.mask = False
    assert not nd.mask
    # Check that it replaces a mask from init
    nd = NDDataBase(nd, mask=True)
    assert nd.mask
    nd.mask = False
    assert not nd.mask


# Init tests
def test_nddata_empty():
    with pytest.raises(TypeError):
        NDDataBase()  # empty initializer should fail


def test_nddata_init_data_nonarray():
    inp = [1, 2, 3]
    nd = NDDataBase(inp)
    assert (np.array(inp) == nd.data).all()


def test_nddata_init_data_ndarray():
    # random floats
    with NumpyRNGContext(123):
        nd = NDDataBase(np.random.random((10, 10)))
    assert nd.data.shape == (10, 10)
    assert nd.data.size == 100
    assert nd.data.dtype == np.dtype(float)

    # specific integers
    nd = NDDataBase(np.array([[1, 2, 3], [4, 5, 6]]))
    assert nd.data.size == 6
    assert nd.data.dtype == np.dtype(int)

    # Tests to ensure that creating a new NDData object copies by *reference*.
    a = np.ones((10, 10))
    nd_ref = NDDataBase(a)
    a[0, 0] = 0
    assert nd_ref.data[0, 0] == 0

    # Except we choose copy=True
    a = np.ones((10, 10))
    nd_ref = NDDataBase(a, copy=True)
    a[0, 0] = 0
    assert nd_ref.data[0, 0] != 0


def test_nddata_init_data_maskedarray():
    with NumpyRNGContext(456):
        NDDataBase(np.random.random((10, 10)),
                   mask=np.random.random((10, 10)) > 0.5)

    # Another test (just copied here)
    with NumpyRNGContext(12345):
        a = np.random.randn(100)
        marr = np.ma.masked_where(a > 0, a)
    nd = NDDataBase(marr)
    # check that masks and data match
    assert_array_equal(nd.mask, marr.mask)
    assert_array_equal(nd.data, marr.data)
    # check that they are both by reference
    marr.mask[10] = ~marr.mask[10]
    marr.data[11] = 123456789
    assert_array_equal(nd.mask, marr.mask)
    assert_array_equal(nd.data, marr.data)

    # or not if we choose copy=True
    nd = NDDataBase(marr, copy=True)
    marr.mask[10] = ~marr.mask[10]
    marr.data[11] = 0
    assert nd.mask[10] != marr.mask[10]
    assert nd.data[11] != marr.data[11]


@pytest.mark.parametrize('data', [np.array([1, 2, 3]), 5])
def test_nddata_init_data_quantity(data):
    # Test an array and a scalar because a scalar Quantity does not always
    # behaves the same way as an array.
    quantity = data * u.adu
    ndd = NDDataBase(quantity)
    assert ndd.unit == quantity.unit
    assert_array_equal(ndd.data, np.array(quantity.value))
    if ndd.data.size > 1:
        # check that if it is an array it is not copied
        quantity.value[1] = 100
        assert ndd.data[1] == quantity.value[1]

        # or is copyied if we choose copy=True
        ndd = NDDataBase(quantity, copy=True)
        quantity.value[1] = 5
        assert ndd.data[1] != quantity.value[1]


def test_nddata_init_data_masked_quantity():
    a = np.array([2, 3])
    q = a * u.m
    m = False
    mq = np.ma.array(q, mask=m)
    nd = NDDataBase(mq)
    assert_array_equal(nd.data, a)
    # This test failed before the change in nddata init because the masked
    # arrays data (which in fact was a quantity was directly saved)
    assert nd.unit == u.m
    assert not isinstance(nd.data, u.Quantity)
    np.testing.assert_array_equal(nd.mask, np.array(m))


def test_nddata_init_data_nddata():
    nd1 = NDDataBase(np.array([1]))
    nd2 = NDDataBase(nd1)
    assert nd2.wcs == nd1.wcs
    assert nd2.uncertainty == nd1.uncertainty
    assert nd2.mask == nd1.mask
    assert nd2.unit == nd1.unit
    assert nd2.meta == nd1.meta

    # Check that it is copied by reference
    nd1 = NDDataBase(np.ones((5, 5)))
    nd2 = NDDataBase(nd1)
    assert nd1.data is nd2.data

    # Check that it is really copied if copy=True
    nd2 = NDDataBase(nd1, copy=True)
    nd1.data[2, 3] = 10
    assert nd1.data[2, 3] != nd2.data[2, 3]

    # Now let's see what happens if we have all explicitly set
    nd1 = NDDataBase(np.array([1]), mask=False, uncertainty=10, unit=u.s,
                     meta={'dest': 'mordor'}, wcs=10, flags=20)
    nd2 = NDDataBase(nd1)
    assert nd2.data is nd1.data
    assert nd2.wcs == nd1.wcs
    assert nd2.uncertainty.data == nd1.uncertainty.data
    assert nd2.mask == nd1.mask
    assert nd2.unit == nd1.unit
    assert nd2.meta == nd1.meta
    assert nd2.flags == nd1.flags

    # now what happens if we overwrite them all too
    nd3 = NDDataBase(nd1, mask=True, uncertainty=200, unit=u.km,
                     meta={'observer': 'ME'}, wcs=4, flags=1)
    assert nd3.data is nd1.data
    assert nd3.wcs != nd1.wcs
    assert nd3.uncertainty.data != nd1.uncertainty.data
    assert nd3.mask != nd1.mask
    assert nd3.unit != nd1.unit
    assert nd3.meta != nd1.meta
    assert nd3.flags != nd1.flags


def test_nddata_init_data_nddata_subclass():
    # There might be some incompatible subclasses of NDData around.
    bnd = BadNDDataSubclass(False, True, 3, 2, 'gollum', 100)
    # Before changing the NDData init this would not have raised an error but
    # would have lead to a compromised nddata instance
    with pytest.raises(TypeError):
        NDDataBase(bnd)
    # but if it has no actual incompatible attributes it passes
    bnd_good = BadNDDataSubclass(np.array([1, 2]), True, 3, 2,
                                 {'enemy': 'black knight'}, u.km)
    nd = NDDataBase(bnd_good)
    assert nd.unit == bnd_good.unit
    assert nd.meta == bnd_good.meta
    assert nd.uncertainty.data == bnd_good.uncertainty
    assert nd.mask == bnd_good.mask
    assert nd.wcs == bnd_good.wcs
    assert nd.data is bnd_good.data


def test_nddata_init_data_fail():
    # First one is slicable but has no shape, so should fail.
    with pytest.raises(TypeError):
        NDDataBase({'a': 'dict'})

    # This has a shape but is not slicable
    class Shape(object):
        def __init__(self):
            self.shape = 5

        def __repr__(self):
            return '7'

    with pytest.raises(TypeError):
        NDDataBase(Shape())


def test_nddata_init_data_fakes():
    ndd1 = NDDataBase(FakeNumpyArray())
    # First make sure that NDData is converting its data to a numpy array.
    assert isinstance(ndd1.data, np.ndarray)
    # Make a new NDData initialized from an NDData
    ndd2 = NDDataBase(ndd1)
    # Check that the data is still an ndarray
    assert isinstance(ndd2.data, np.ndarray)


def test_param_default_sentinels():
    quantity = np.arange(10) * u.m
    ndd = NDDataBase(quantity, unit=None)
    assert ndd.unit is None

    masked_array = np.ma.array(np.ones(3), mask=[1, 0, 1])
    ndd = NDDataBase(masked_array, mask=None)
    assert ndd.mask is None

    nddata_all = NDDataBase(100, mask=True, uncertainty=5, unit='cm', wcs=10,
                            meta={1: 10})
    ndd = NDDataBase(nddata_all, mask=None, uncertainty=None, unit=None,
                     wcs=None, meta=None)
    assert ndd.mask is None
    assert ndd.uncertainty is None
    assert ndd.unit is None
    assert ndd.wcs is None
    assert len(ndd.meta) == 0
    assert isinstance(ndd.meta, OrderedDict)


# Specific parameters
def test_param_uncertainty():
    u = StdDevUncertainty(np.ones((5, 5)))
    d = NDDataBase(np.ones((5, 5)), uncertainty=u)
    # Test that the parent_nddata is set.
    assert d.uncertainty.parent_nddata is d
    # Test conflicting uncertainties (other NDData)
    u2 = StdDevUncertainty(np.ones((5, 5))*2)
    d2 = NDDataBase(d, uncertainty=u2)
    assert d2.uncertainty is u2
    assert d2.uncertainty.parent_nddata is d2


def test_param_wcs():
    # Since everything is allowed we only need to test something
    nd = NDDataBase([1], wcs=3)
    assert nd.wcs == 3
    # Test conflicting wcs (other NDData)
    nd2 = NDDataBase(nd, wcs=2)
    assert nd2.wcs == 2


def test_param_meta():
    # everything dict-like is allowed
    with pytest.raises(TypeError):
        NDDataBase([1], meta=3)
    nd = NDDataBase([1, 2, 3], meta={})
    assert len(nd.meta) == 0
    nd = NDDataBase([1, 2, 3])
    assert isinstance(nd.meta, OrderedDict)
    assert len(nd.meta) == 0
    # Test conflicting meta (other NDData)
    nd2 = NDDataBase(nd, meta={'image': 'sun'})
    assert len(nd2.meta) == 1
    nd3 = NDDataBase(nd2, meta={'image': 'moon'})
    assert len(nd3.meta) == 1
    assert nd3.meta['image'] == 'moon'


def test_param_meta_copied():
    # This is more a regression test after I switched from the
    # astropy.utils.metadata.MetaData descriptor to the custom one I created.
    meta = {1: 1}
    # also test data=None ... that's new :-)
    ndd = NDDataBase(None, meta=meta, copy=True)
    ndd.meta[1] = 2
    assert meta[1] == 1

    ndd = NDDataBase(None, meta=meta)
    ndd.meta[1] = 2
    assert meta[1] == 2


def test_param_mask():
    # Since everything is allowed we only need to test something
    nd = NDDataBase([1], mask=False)
    assert not nd.mask
    # Test conflicting mask (other NDData)
    nd2 = NDDataBase(nd, mask=True)
    assert nd2.mask
    # (masked array)
    nd3 = NDDataBase(np.ma.array([1], mask=False), mask=True)
    assert nd3.mask
    # (masked quantity)
    mq = np.ma.array(np.array([2, 3])*u.m, mask=False)
    nd4 = NDDataBase(mq, mask=True)
    assert nd4.mask


def test_param_unit():
    with pytest.raises(ValueError):
        NDDataBase(np.ones((5, 5)), unit="NotAValidUnit")
    NDDataBase([1, 2, 3], unit='meter')
    # Test conflicting units (quantity as data)
    q = np.array([1, 2, 3]) * u.m
    nd = NDDataBase(q, unit='cm')
    assert nd.unit != q.unit
    assert nd.unit == u.cm
    # (masked quantity)
    mq = np.ma.array(np.array([2, 3])*u.m, mask=False)
    nd2 = NDDataBase(mq, unit=u.s)
    assert nd2.unit == u.s
    # (another NDData as data)
    nd3 = NDDataBase(nd, unit='km')
    assert nd3.unit == u.km


def test_param_flags():
    flags = np.ones((20, 20))
    ndd = NDDataBase(None, flags=flags)
    assert ndd.flags is not None
    assert_array_equal(flags, ndd.flags)

    ndd2 = NDDataBase(ndd, copy=True)
    assert ndd.flags is not None
    assert_array_equal(flags, ndd.flags)

    # Test if changes are propagated to uncopied attributes
    flags[0, 0] = 100
    assert ndd.flags[0, 0] == 100
    assert ndd2.flags[0, 0] == 1


# Check that the meta descriptor is working as expected. The MetaBaseTest class
# takes care of defining all the tests, and we simply have to define the class
# and any minimal set of args to pass.
from astropy.utils.tests.test_metadata import MetaBaseTest


class TestMetaNDData(MetaBaseTest):
    test_class = NDDataBase
    args = np.array([[1.]])


# Representation tests
def test_nddata_repr():
    arr1d = NDDataBase(np.array([1, 2, 3]))
    assert repr(arr1d) == 'NDDataBase([1, 2, 3])'

    arr2d = NDDataBase(np.array([[1, 2], [3, 4]]))
    assert repr(arr2d) == textwrap.dedent("""
        NDDataBase([[1, 2],
                    [3, 4]])"""[1:])

    arr3d = NDDataBase(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
    assert repr(arr3d) == textwrap.dedent("""
        NDDataBase([[[1, 2],
                     [3, 4]],

                    [[5, 6],
                     [7, 8]]])"""[1:])


def test_nddata_str():
    # We know that the representation works, so just compare it to the repr:
    arr1d = NDDataBase(np.array([1, 2, 3]))
    assert str(arr1d) == repr(arr1d)

    arr2d = NDDataBase(np.array([[1, 2], [3, 4]]))
    assert str(arr2d) == repr(arr2d)

    arr3d = NDDataBase(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
    assert str(arr3d) == repr(arr3d)


def test_nddata_interface():
    nddlike = NDDataInterface([1], 2, 3, 4, {1: 1}, 6, 7)
    ndd = NDDataBase(nddlike)

    # TODO: akward test because comparing number to Quantity
    assert ndd.unit == nddlike._unit
    assert ndd.mask == nddlike._mask
    assert ndd.wcs == nddlike._wcs
    assert ndd.meta == nddlike._meta
    assert type(ndd.meta) == type(nddlike._meta) # make sure it's really a dict
    np.testing.assert_array_equal(ndd.data, np.asarray(nddlike._data))
    np.testing.assert_array_equal(ndd.uncertainty.data,
                                  np.asarray(nddlike._uncertainty))
    assert isinstance(ndd.uncertainty, UnknownUncertainty)
    assert ndd.flags == 7


def test_nddata_partial_interface():
    nddlike = NDDataPartialInterface([1], 2, 3, 4, {1: 1}, 6)
    ndd = NDDataBase(nddlike)

    # TODO: akward test because comparing number to Quantity
    assert ndd.unit == nddlike._unit
    assert ndd.meta == nddlike._meta
    assert type(ndd.meta) == type(nddlike._meta) # make sure it's really a dict
    np.testing.assert_array_equal(ndd.data, np.asarray(nddlike._data))
    assert ndd.uncertainty is None
    assert ndd.mask is None
    assert ndd.wcs is None


def test_nddata_no_data_interface():
    nddlike = NDDataBrokenInterface([1], 2, 3, 4, {1: 1}, 6)
    ndd = NDDataBase(nddlike)

    # TODO: akward test because comparing number to Quantity
    assert ndd.unit == nddlike._unit
    assert ndd.mask == nddlike._mask
    assert ndd.wcs == nddlike._wcs
    assert ndd.meta == nddlike._meta
    assert type(ndd.meta) == type(nddlike._meta) # make sure it's really a dict
    assert ndd.data is None
    np.testing.assert_array_equal(ndd.uncertainty.data,
                                  np.asarray(nddlike._uncertainty))
    assert isinstance(ndd.uncertainty, UnknownUncertainty)


def test_copy_direct():
    ndd = NDDataBase(np.ones((3, 3)), mask=np.ones((3, 3)), unit='m',
                     meta={'a': 100}, uncertainty=np.ones((3, 3)),
                     wcs=np.ones((3, 3)), flags=np.ones((3, 3)))

    ndd2 = ndd.copy()

    # Alter elements so we can verify if copied
    ndd.data[0, 0] = 10
    ndd.mask[0, 0] = 0
    ndd.meta['a'] = 10
    ndd.uncertainty.data[0, 0] = 10
    ndd.wcs[0, 0] = 10
    ndd.flags[0, 0] = 10

    assert ndd2.data[0, 0] == 1
    assert ndd2.mask[0, 0] == 1
    assert ndd2.meta['a'] == 100
    assert ndd2.uncertainty.data[0, 0] == 1
    assert ndd2.wcs[0, 0] == 1
    assert ndd2.flags[0, 0] == 1

    # Check if the uncertainties link to the right parent.
    assert ndd2.uncertainty.parent_nddata is ndd2
    assert ndd.uncertainty.parent_nddata is ndd


def test_copy_indirect():
    from copy import copy, deepcopy

    # Both functions should return a deepcopy.
    for copyfunc in (copy, deepcopy):
        ndd = NDDataBase(np.ones((3, 3)), mask=np.ones((3, 3)), unit='m',
                         meta={'a': 100}, uncertainty=np.ones((3, 3)),
                         wcs=np.ones((3, 3)), flags=np.ones((3, 3)))

        ndd2 = copyfunc(ndd)

        # Alter elements so we can verify if copied
        ndd.data[0, 0] = 10
        ndd.mask[0, 0] = 0
        ndd.meta['a'] = 10
        ndd.uncertainty.data[0, 0] = 10
        ndd.wcs[0, 0] = 10
        ndd.flags[0, 0] = 10

        assert ndd2.data[0, 0] == 1
        assert ndd2.mask[0, 0] == 1
        assert ndd2.meta['a'] == 100
        assert ndd2.uncertainty.data[0, 0] == 1
        assert ndd2.wcs[0, 0] == 1
        assert ndd2.flags[0, 0] == 1

        # Check if the uncertainties link to the right parent.
        assert ndd2.uncertainty.parent_nddata is ndd2
        assert ndd.uncertainty.parent_nddata is ndd


# Not supported features
def test_slicing_not_supported():
    ndd = NDDataBase(np.ones((5, 5)))
    with pytest.raises(TypeError):
        ndd[0]


def test_arithmetic_not_supported():
    ndd = NDDataBase(np.ones((5, 5)))
    with pytest.raises(TypeError):
        ndd + ndd


def test_mask_as_boolean_mask():
    ndd = NDDataBase(np.ones((3, 3)), mask=False)
    with pytest.raises(ValueError):
        ndd._get_mask_numpylike()

    ndd.mask = np.ones((3, 3))
    assert ndd._get_mask_numpylike().dtype == bool
