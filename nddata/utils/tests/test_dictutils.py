# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import math

from astropy.tests.helper import pytest

from ..dictutils import dict_split, dict_merge

# Just tests that ensure the appropriate exceptions are returned:


def test_split_fails():
    # Splitting fails if some value doesn't contain the seperator
    adict = {'a': 10}
    with pytest.raises(ValueError):
        dict_split(adict)


def test_merge_fails():
    # Merging fails if the foldfunc is not recognized
    adict1 = {'a': 10}
    adict2 = {'a': 20}
    with pytest.raises(ValueError):
        dict_merge(adict1, adict2, foldfunc='funny')

    # or the foldfunc takes not two arguments
    with pytest.raises(TypeError):
        dict_merge(adict1, adict2, foldfunc=math.sqrt)
