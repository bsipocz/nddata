# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from .. import is_numeric


def test_not_array():
    assert is_numeric(1)
    assert is_numeric(1.)
    assert is_numeric(1+1j)
    assert not is_numeric('a')
    assert not is_numeric(None)
    assert is_numeric([1, 2, 3])


def test_array():
    assert is_numeric(np.array(1))
    assert is_numeric(np.array(1.))
    assert is_numeric(np.array(1+1j))
    assert is_numeric(np.array([1]))
    assert is_numeric(np.array([1.]))
    assert is_numeric(np.array([1+1j]))
    assert not is_numeric(np.array('a'))
    assert not is_numeric(np.array(['a']))
