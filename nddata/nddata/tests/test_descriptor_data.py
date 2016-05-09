# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
import numpy as np

import astropy.units as u

from .. import DataDescriptor


class ClassForTests(object):
    data = DataDescriptor()


def test_data_valid_types_scalar():
    obj = ClassForTests()
    # Scalar
    obj.data = 1


def test_data_valid_types_list():
    obj = ClassForTests()
    # List integer
    alist = [1, 2, 3]
    obj.data = alist
    np.testing.assert_array_equal(obj.data, np.array(alist))
    obj.data[1] = 10
    assert alist[1] == 2  # it's saved as copy


def test_data_valid_types_tuple():
    obj = ClassForTests()
    # Tuple floats
    atuple = (1., 2., 3.)
    obj.data = atuple
    np.testing.assert_array_equal(obj.data, np.array(atuple))
    obj.data[1] = 10
    assert atuple[1] == 2  # it's saved as copy


def test_data_valid_types_None():
    obj = ClassForTests()
    # None
    obj.data = None
    assert obj.data is None


def test_data_valid_types_numpy_array():
    obj = ClassForTests()
    # numpy array
    anarray = np.ones((10, 10), dtype=complex)
    obj.data = anarray
    np.testing.assert_array_equal(obj.data, anarray)
    obj.data[1, 3] = 10
    assert anarray[1, 3] == 10  # it's saved as reference


def test_data_valid_types_numpy_masked_array():
    obj = ClassForTests()
    # numpy masked array loses it's mask
    amaskedarray = np.ma.array(np.ones((10, 10), dtype=float), mask=True)
    obj.data = amaskedarray
    np.testing.assert_array_equal(amaskedarray.data, obj.data)
    assert getattr(obj.data, 'mask', None) is None
    obj.data[4, 4] = 200
    assert amaskedarray.data[4, 4] == 200


def test_data_valid_types_astropy_quantity():
    obj = ClassForTests()
    # Quantity will lose it's unit
    aquantity = np.ones((10, 10), dtype=float) * u.m
    obj.data = aquantity
    np.testing.assert_array_equal(aquantity.value, obj.data)
    assert getattr(obj.data, 'unit', None) is None
    obj.data[4, 4] = 200
    assert aquantity.value[4, 4] == 200


def test_data_invalid_types():
    obj = ClassForTests()

    # String
    with pytest.raises(TypeError):
        obj.data = 'hello'

    # Mixed list
    with pytest.raises(TypeError):
        obj.data = [1, 2, None, 4]
