# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import math

from astropy.tests.helper import pytest

from ..dictutils import (dict_split, dict_merge, dict_merge_keep_all,
                         dict_merge_keep_all_fill_missing)

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


def test_merge_nodict():
    assert not dict_merge()
    assert not dict_merge_keep_all()
    assert not dict_merge_keep_all_fill_missing()


def test_merge_keep_all():
    dict1 = {'a': 1, 'b': 1, 'c': 1}
    result = dict_merge_keep_all(dict1)
    assert result == {'a': [1], 'b': [1], 'c': [1]}

    dict1 = {'a': 1, 'b': 1, 'c': 1, 'd': 1}
    dict2 = {'a': 2, 'b': 2, 'c': 2, 'e': 2}
    dict3 = {'a': 3, 'b': 3, 'c': 3, 'f': 3}
    merged = dict_merge_keep_all(dict1, dict2, dict3)
    reference = {'a': [1, 2, 3], 'b': [1, 2, 3], 'c': [1, 2, 3], 'd': [1],
                 'e': [2], 'f': [3]}
    assert merged == reference


def test_merge_keep_all_fill_missing():
    dict1 = {'a': 1, 'b': 1, 'c': 1}
    result = dict_merge_keep_all_fill_missing(dict1)
    assert result == {'a': [1], 'b': [1], 'c': [1]}

    dict1 = {'a': 1, 'b': 1, 'c': 1, 'd': 1}
    dict2 = {'a': 2, 'b': 2, 'c': 2, 'e': 2}
    dict3 = {'a': 3, 'b': 3, 'c': 3, 'f': 3}
    merged = dict_merge_keep_all_fill_missing(dict1, dict2, dict3)
    reference = {'a': [1, 2, 3], 'b': [1, 2, 3],
                 'c': [1, 2, 3], 'd': [1, None, None],
                 'e': [None, 2, None], 'f': [None, None, 3]}
    assert merged == reference

    dict1 = {'a': 1, 'b': 1, 'c': 1, 'd': 1}
    dict2 = {'a': 2, 'b': 2, 'c': 2, 'e': 2}
    dict3 = {'a': 3, 'b': 3, 'c': 3, 'f': 3}
    merged = dict_merge_keep_all_fill_missing(dict1, dict2, dict3, fill=0)
    reference = {'a': [1, 2, 3], 'b': [1, 2, 3], 'c': [1, 2, 3],
                 'd': [1, 0, 0], 'e': [0, 2, 0], 'f': [0, 0, 3]}
    assert merged == reference
