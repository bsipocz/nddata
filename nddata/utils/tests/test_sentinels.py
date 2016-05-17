# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from copy import copy, deepcopy

from astropy.tests.helper import pytest

from ..sentinels import SentinelFactory


def test_create_sentinel():
    # Create two sentinels
    TestSentinel1 = SentinelFactory('test1')
    TestSentinel2 = SentinelFactory('test2')

    # fails when given a not-string
    with pytest.raises(TypeError):
        SentinelFactory(1)

    # different sentinals are not the same (trivial but should be tested)
    assert TestSentinel1 is not TestSentinel2

    # any string representation returns the initial string
    assert repr(TestSentinel1) == 'test1'
    assert str(TestSentinel2) == 'test2'

    # Copy doesn't copy but returns the same instance
    copy1 = copy(TestSentinel1)
    copy2 = deepcopy(TestSentinel1)
    assert copy1 is TestSentinel1
    assert copy2 is TestSentinel1

    # Attributes are not settable or deletable
    with pytest.raises(TypeError):
        TestSentinel1.name = 'test'

    with pytest.raises(TypeError):
        del TestSentinel1.name

    # They evaluate to False in ifs:
    assert not TestSentinel1
    assert not TestSentinel2
    assert not TestSentinel1.__bool__()    # explicit call for python 3 magic
