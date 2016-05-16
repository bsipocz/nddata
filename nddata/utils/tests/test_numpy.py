# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from ..numpy import is_numeric_array


NUMERIC = [True, 1, -1, 1.0, 1+1j]
NOT_NUMERIC = [object(), 'string', u'unicode', None]


def test_is_numeric():

    for x in NUMERIC:
        for y in (x, [x], [x] * 2):
            for z in (y, np.array(y)):
                assert is_numeric_array(z) is True

    for x in NOT_NUMERIC:
        for y in (x, [x], [x] * 2):
            for z in (y, np.array(y)):
                assert is_numeric_array(z) is False

    for kind, dtypes in np.sctypes.items():
        if kind != 'others':
            for dtype in dtypes:
                assert is_numeric_array(np.array([0], dtype=dtype)) is True
