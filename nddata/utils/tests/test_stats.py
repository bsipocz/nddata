# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from ..stats import mode
from ..stats import NUMPY_1_9


def test_mode():
    # General case
    assert mode([1., 1., 2., 3., 4., 5.]) == (1.0, 2)
    # Returns the smallest if several are most common
    assert mode([1., 2., 3., 4., 5.]) == (1.0, 1)
    # Flattens the array
    assert mode([[1., 1.], [2., 3.], [4., 5.]]) == (1.0, 2)
    # Rounds down
    assert mode([0.1, 0.1, 0.2]) == (0.0, 3)


def test_mode_decimals():

    # This should be the same for both cases.
    assert mode([0.5, 0.5, 0.7]) == (0.0, 2)  # 0.5 rounds down
    assert mode([1.5, 1.5, 1.7]) == (2.0, 3)  # 1.5 rounds up

    if NUMPY_1_9:
        assert mode([0.5, 0.5, 0.7], decimals=1) == (0.5, 2)
        assert mode([0.5, 0.5, 0.7], decimals=None) == (0.5, 2)
        assert mode([1.5, 1.5, 1.7], decimals=1) == (1.5, 2)
    else:
        # NumPy 1.8 and earlier ignores decimals.
        assert mode([0.5, 0.5, 0.7], decimals=1) == (0.0, 2)  # like without
        assert mode([1.5, 1.5, 1.7], decimals=1) == (2.0, 3)  # like without
