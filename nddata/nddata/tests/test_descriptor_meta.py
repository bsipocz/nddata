# Licensed under a 3-clause BSD style license - see LICENSE.rst

import collections
import pytest

from astropy.io.fits import Header

from .. import MetaDescriptor


class ClassForTests(object):
    meta = MetaDescriptor()

    def validate_meta_is_always_mapping(self):
        assert isinstance(self.meta, collections.Mapping)

    def validate_has_x_elements(self, number=0):
        assert len(self.meta) == number


def test_meta_valid_empty_inputs():
    # Create a class and don't set the attribute and check if the return
    obj = ClassForTests()
    obj.validate_meta_is_always_mapping()
    obj.validate_has_x_elements()

    obj.meta = {}
    obj.validate_meta_is_always_mapping()
    obj.validate_has_x_elements()

    obj.meta = collections.defaultdict(int)
    obj.validate_meta_is_always_mapping()
    obj.validate_has_x_elements()

    obj.meta = Header()
    obj.validate_meta_is_always_mapping()
    obj.validate_has_x_elements()

    obj.meta = None
    obj.validate_meta_is_always_mapping()
    obj.validate_has_x_elements()


def test_meta_saved_as_reference():
    obj = ClassForTests()
    meta = {1: 2}
    obj.meta = meta

    obj.meta[1] = 10
    assert obj.meta == meta

    obj.validate_meta_is_always_mapping()
    obj.validate_has_x_elements(1)

    obj.meta['a'] = 20
    assert obj.meta == meta

    obj.validate_meta_is_always_mapping()
    obj.validate_has_x_elements(2)


def test_meta_overwrite():
    obj = ClassForTests()
    obj.meta = {1: 1}

    obj.validate_meta_is_always_mapping()
    obj.validate_has_x_elements(1)

    obj.meta = {2: 2}

    obj.validate_meta_is_always_mapping()
    obj.validate_has_x_elements(1)


def test_meta_setter_exception():
    obj = ClassForTests()

    # Give an invalid input
    with pytest.raises(TypeError):
        obj.meta = 10

    obj.validate_meta_is_always_mapping()
    obj.validate_has_x_elements()

    # Now check that an existing meta isn't deleted when invalid inputs are
    # given.

    obj.meta = {1: 2}
    obj.validate_meta_is_always_mapping()
    obj.validate_has_x_elements(1)

    with pytest.raises(TypeError):
        obj.meta = 10

    assert obj.meta == {1: 2}
