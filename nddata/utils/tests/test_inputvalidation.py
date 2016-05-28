# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from astropy.tests.helper import pytest

from ..inputvalidation import as_integer, as_unsigned_integer


def test_is_integer():
    # not convertable to integer like a floating STRING
    with pytest.raises(ValueError):
        as_integer('3.14')
    # or a list
    with pytest.raises(ValueError):
        as_integer([1, 2])

    # Test regular cases:
    for i in [-10, -5, 0, 2, 1000]:
        assert as_integer(i) == i

    # Test values that have another type but can be converted to integer and
    # remain the same:
    for i in [-2.0, 0.0]:
        for info in [True, False]:
            assert as_integer(i, info=info) == i

    # Test that it fails with info=False and the value differs from an integer
    for i in [-1.2, 2.1]:
        with pytest.raises(ValueError):
            as_integer(i, info=False)

    # But it's converted to a plain integer with info=True
    for i in [-1.2, 2.1]:
        assert as_integer(i, info=True) == int(i)


def test_is_unsignedinteger():
    # not convertable to integer like a floating STRING
    with pytest.raises(ValueError):
        as_unsigned_integer('3.14')
    # or a list
    with pytest.raises(ValueError):
        as_unsigned_integer([1, 2])

    # Test regular cases:
    for i in [0, 2, 1000]:
        assert as_unsigned_integer(i) == i

    # Test values that have another type but can be converted to integer and
    # remain the same:
    for i in [0.0, 2.0]:
        for info in [True, False]:
            assert as_unsigned_integer(i, info=info) == i

    # Test that it fails with info=False and the value differs from an integer
    # or the value is a negative integer.
    for i in [1.1, 2.1, -1, -1.2]:
        with pytest.raises(ValueError):
            as_unsigned_integer(i, info=False)

    # But it's converted to the absolute and a plain integer with info=True
    for i in [-1.2, 2.1]:
        assert as_unsigned_integer(i, info=True) == abs(int(i))
