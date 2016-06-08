# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import absolute_import, division, print_function

from itertools import tee

from astropy.tests.helper import pytest

from ..itertools_recipes import tee_lookahead


def test_tee_lookahead():
    t1, t2 = tee([1, 2, 3, 4, 5])
    # Just a test for the IndexError. Everything else should be tested with the
    # doctests.
    with pytest.raises(IndexError):
        tee_lookahead(t1, 10)
