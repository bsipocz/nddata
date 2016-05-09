# Licensed under a 3-clause BSD style license - see LICENSE.rst

import weakref

import pytest
import numpy as np

from .. import ParentNDDataDescriptor
from .. import MissingParentNDDataException


class ClassForTests(object):
    parent_nddata = ParentNDDataDescriptor()


def test_parent_nddata_none_set():
    # Create a class and don't set the attribute and check if the return
    obj = ClassForTests()
    with pytest.raises(MissingParentNDDataException):
        obj.parent_nddata


def test_parent_nddata_unset():
    # Create a class and don't set the attribute and check if the return
    obj = ClassForTests()
    parent = np.ones((10, 10))
    obj.parent_nddata = parent

    # private attribute
    assert isinstance(obj._parent_nddata, weakref.ref)
    # public attribute
    np.testing.assert_array_equal(obj.parent_nddata, parent)

    # after unsetting it it should raise the Exception again.
    obj.parent_nddata = None
    with pytest.raises(MissingParentNDDataException):
        obj.parent_nddata


def test_inject_non_weakref_private():
    obj = ClassForTests()
    parent = np.ones((10, 10))
    obj._parent_nddata = parent  # setting the private attribute!

    with pytest.raises(TypeError):
        obj.parent_nddata


def test_saves_as_reference():
    # Check that it really saves it as reference.
    obj = ClassForTests()
    parent = np.ones((10, 10))
    obj.parent_nddata = parent

    obj.parent_nddata[3, 3] = 200

    assert parent[3, 3] == 200
