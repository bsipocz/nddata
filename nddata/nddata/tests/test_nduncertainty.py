# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from numpy.testing import assert_array_equal

from astropy.tests.helper import pytest
from astropy import units as u

from ..nduncertainty_stddev import StdDevUncertainty
from ..nduncertainty_unknown import UnknownUncertainty
from ..meta import NDUncertainty
from ..exceptions import IncompatibleUncertaintiesException
from ..exceptions import MissingDataAssociationException
from ..nddata import NDDataBase

from ...utils.garbagecollector import assert_memory_leak

# Regarding setter tests:
# No need to test setters since the uncertainty is considered immutable after
# creation except of the parent_nddata attribute and this accepts just
# everything.
# Additionally they should be covered by NDData, NDArithmeticMixin which rely
# on it

# Regarding propagate, _convert_uncert, _propagate_* tests:
# They should be covered by NDArithmeticMixin since there is generally no need
# to test them without this mixin.

# Regarding __getitem__ tests:
# Should be covered by NDSlicingMixin.

# Regarding StdDevUncertainty tests:
# This subclass only overrides the methods for propagation so the same
# they should be covered in NDArithmeticMixin.

# Not really fake but the minimum an uncertainty has to override not to be
# abstract.


class FakeUncertainty(NDUncertainty):

    @property
    def uncertainty_type(self):
        return 'fake'

# Test the fake (added also StdDevUncertainty which should behave identical)


@pytest.mark.parametrize(('UncertClass'), [FakeUncertainty, StdDevUncertainty,
                                           UnknownUncertainty])
def test_init_fake_with_list(UncertClass):
    fake_uncert = UncertClass([1, 2, 3])
    assert_array_equal(fake_uncert.data, np.array([1, 2, 3]))
    # Copy makes no difference since casting a list to an np.ndarray always
    # makes a copy.
    # But let's give the uncertainty a unit too
    fake_uncert = UncertClass([1, 2, 3], unit=u.adu)
    assert_array_equal(fake_uncert.data, np.array([1, 2, 3]))
    assert fake_uncert.unit is u.adu


@pytest.mark.parametrize(('UncertClass'), [FakeUncertainty, StdDevUncertainty,
                                           UnknownUncertainty])
def test_init_fake_with_ndarray(UncertClass):
    uncert = np.arange(100).reshape(10, 10)
    fake_uncert = UncertClass(uncert, copy=True)
    # Numpy Arrays are copied by default
    assert_array_equal(fake_uncert.data, uncert)
    assert fake_uncert.data is not uncert
    # Now try it without copy
    fake_uncert = UncertClass(uncert, copy=False)
    assert fake_uncert.data is uncert
    # let's provide a unit
    fake_uncert = UncertClass(uncert, unit=u.adu, copy=True)
    assert_array_equal(fake_uncert.data, uncert)
    assert fake_uncert.data is not uncert
    assert fake_uncert.unit is u.adu


@pytest.mark.parametrize(('UncertClass'), [FakeUncertainty, StdDevUncertainty,
                                           UnknownUncertainty])
def test_init_fake_with_quantity(UncertClass):
    uncert = np.arange(10).reshape(2, 5) * u.adu
    # with copy
    fake_uncert = UncertClass(uncert, copy=True)
    assert_array_equal(fake_uncert.data, uncert.value)
    assert fake_uncert.data is not uncert.value
    assert fake_uncert.unit is u.adu
    # Try without copy (should not work, quantity.value always returns a copy)
    fake_uncert = UncertClass(uncert, copy=False)
    assert fake_uncert.data is not uncert.value
    assert fake_uncert.unit is u.adu
    # Now try with an explicit unit parameter too
    fake_uncert = UncertClass(uncert, unit=u.m, copy=True)
    assert_array_equal(fake_uncert.data, uncert.value)  # No conversion done
    assert fake_uncert.data is not uncert.value
    assert fake_uncert.unit is u.m  # It took the explicit one


@pytest.mark.parametrize(('UncertClass'), [FakeUncertainty, StdDevUncertainty,
                                           UnknownUncertainty])
def test_init_fake_with_fake(UncertClass):
    uncert = np.arange(5).reshape(5, 1)
    fake_uncert1 = UncertClass(uncert, copy=True)
    fake_uncert2 = UncertClass(fake_uncert1, copy=True)
    assert_array_equal(fake_uncert2.data, uncert)
    assert fake_uncert2.data is not uncert
    # Without making copies
    fake_uncert1 = UncertClass(uncert, copy=False)
    fake_uncert2 = UncertClass(fake_uncert1, copy=False)
    assert_array_equal(fake_uncert2.data, fake_uncert1.data)
    assert fake_uncert2.data is fake_uncert1.data
    # With a unit
    uncert = np.arange(5).reshape(5, 1) * u.adu
    fake_uncert1 = UncertClass(uncert, copy=True)
    fake_uncert2 = UncertClass(fake_uncert1, copy=True)
    assert_array_equal(fake_uncert2.data, uncert.value)
    assert fake_uncert2.data is not uncert.value
    assert fake_uncert2.unit is u.adu
    # With a unit and an explicit unit-parameter
    fake_uncert2 = UncertClass(fake_uncert1, unit=u.cm, copy=True)
    assert_array_equal(fake_uncert2.data, uncert.value)
    assert fake_uncert2.data is not uncert.value
    assert fake_uncert2.unit is u.cm


# This test explicitly NOT works with StdDevUncertainty since it would always
# convert to a numpy array.
@pytest.mark.parametrize(('UncertClass'), [FakeUncertainty,
                                           UnknownUncertainty])
def test_init_fake_with_somethingElse(UncertClass):
    # What about a dict?
    uncert = {'rdnoise': 2.9, 'gain': 0.6}
    fake_uncert = UncertClass(uncert)
    assert fake_uncert.data == uncert
    # We can pass a unit too but since we cannot do uncertainty propagation
    # the interpretation is up to the user
    fake_uncert = UncertClass(uncert, unit=u.s)
    assert fake_uncert.data == uncert
    assert fake_uncert.unit is u.s
    # So, now check what happens if copy is False
    fake_uncert = UncertClass(uncert, copy=False)
    assert fake_uncert.data == uncert
    assert id(fake_uncert) != id(uncert)
    # dicts cannot be referenced without copy
    # TODO : Find something that can be referenced without copy :-)


def test_init_fake_with_StdDevUncertainty():
    # Different instances of uncertainties are not directly convertible so this
    # should fail
    uncert = np.arange(5).reshape(5, 1)
    std_uncert = StdDevUncertainty(uncert)
    with pytest.raises(IncompatibleUncertaintiesException):
        FakeUncertainty(std_uncert)
    # Ok try it the other way around
    fake_uncert = FakeUncertainty(uncert)
    with pytest.raises(IncompatibleUncertaintiesException):
        StdDevUncertainty(fake_uncert)


def test_uncertainty_type():
    fake_uncert = FakeUncertainty([10, 2])
    assert fake_uncert.uncertainty_type == 'fake'
    std_uncert = StdDevUncertainty([10, 2])
    assert std_uncert.uncertainty_type == 'std'


def test_uncertainty_correlated():
    std_uncert = StdDevUncertainty([10, 2])
    assert std_uncert.supports_correlated


def test_for_leak_with_uncertainty():
    # Regression test for memory leak because of cyclic references between
    # NDData and uncertainty

    def non_leaker():
        NDDataBase(np.ones(100))

    def leaker():
        NDDataBase(np.ones(100), uncertainty=StdDevUncertainty(np.ones(100)))

    assert_memory_leak(non_leaker, NDDataBase)
    assert_memory_leak(leaker, NDDataBase)


def test_for_stolen_uncertainty():
    # Sharing uncertainties should not overwrite the parent_nddata attribute
    ndd1 = NDDataBase(1, uncertainty=1)
    ndd2 = NDDataBase(2, uncertainty=ndd1.uncertainty)
    # uncertainty.parent_nddata.data should be the original data!
    assert ndd1.uncertainty.parent_nddata.data == ndd1.data

    # just check if it worked also for ndd2
    assert ndd2.uncertainty.parent_nddata.data == ndd2.data


def test_repr():
    uncert = StdDevUncertainty([1, 2, 3])
    assert str(uncert) == 'StdDevUncertainty([1, 2, 3])'
    assert uncert.__repr__() == str(uncert)


def test_param_parent():
    ndd1 = NDDataBase([1, 2, 3], StdDevUncertainty(None))
    ndd2 = NDDataBase(ndd1, copy=True)

    # Not given so it should link to the same parent as the first argument.
    uncert = StdDevUncertainty(ndd1.uncertainty)
    assert uncert.parent_nddata is ndd1

    # Explicitly given will overwrite the current parent so it should link to
    # the other one.
    uncert = StdDevUncertainty(ndd1.uncertainty, parent_nddata=ndd2)
    assert uncert.parent_nddata is ndd2


def test_mess_with_private_parent():
    # DON'T DO THIS ... EVER
    ndd1 = NDDataBase([1, 2, 3], StdDevUncertainty(None))
    ndd1.uncertainty._parent_nddata = ndd1  # direct link leads to memory leak!

    # TODO: This will raise a Warning. Catch it.
    assert ndd1.uncertainty.parent_nddata is ndd1

    # To avoid keeping this as memory leak delete both links manually
    ndd1.uncertainty._parent_nddata = None
    ndd1._uncertainty = None


def test_copy():
    from copy import copy, deepcopy

    parentnddata = NDDataBase([1])

    for unit in ['m', None]:
        for parent in [parentnddata, None]:

            uncert_data = np.array([1, 2, 3])

            uncertainty = StdDevUncertainty(uncert_data, unit=unit,
                                            parent_nddata=parent)
            copy1 = uncertainty.copy()
            copy2 = copy(uncertainty)
            copy3 = deepcopy(uncertainty)
            copy4 = StdDevUncertainty(uncertainty, copy=True)

            uncert_data[0] = 5

            # Check that the uncertainties are unaffected
            assert uncertainty.data[0] == 5
            assert copy1.data[0] == 1
            assert copy2.data[0] == 1
            assert copy3.data[0] == 1
            assert copy4.data[0] == 1

            assert copy1.unit == uncertainty.unit
            assert copy2.unit == uncertainty.unit
            assert copy3.unit == uncertainty.unit
            assert copy4.unit == uncertainty.unit

            if parent is None:
                with pytest.raises(MissingDataAssociationException):
                    uncertainty.parent_nddata
                with pytest.raises(MissingDataAssociationException):
                    copy1.parent_nddata
                with pytest.raises(MissingDataAssociationException):
                    copy2.parent_nddata
                with pytest.raises(MissingDataAssociationException):
                    copy3.parent_nddata
                with pytest.raises(MissingDataAssociationException):
                    copy4.parent_nddata
            else:
                assert uncertainty.parent_nddata is parentnddata
                assert copy1.parent_nddata is parentnddata
                assert copy2.parent_nddata is parentnddata
                assert copy3.parent_nddata is parentnddata
                assert copy4.parent_nddata is parentnddata
