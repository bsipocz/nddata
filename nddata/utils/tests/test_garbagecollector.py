# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from astropy.tests.helper import pytest

from ..garbagecollector import assert_memory_leak

def test_memory_leak():
    def func():
        pass

    assert_memory_leak(func)

    assert_memory_leak(func, dict)


def test_memory_leak_true():
    class Top(object):
        pass

    def func():
        a = Top()
        a.data = np.ones((3, 3))
        b = Top()
        b.data = np.ones((3, 3))
        a.link = b
        b.link = a

    with pytest.raises(AssertionError):
        assert_memory_leak(func)

    with pytest.raises(AssertionError):
        assert_memory_leak(func, Top)

    assert_memory_leak(func, int)
