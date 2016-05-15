# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from collections import OrderedDict

from astropy.tests.helper import pytest
from astropy.io.fits import Header

from ..descriptors import MetaData


class MetaWithoutDocWithCopy(object):
    # descriptor without doc and copy=True
    meta = MetaData()


class MetaWithDocWithoutCopy(object):
    # descriptor with doc and copy=False
    meta = MetaData(doc='test', copy=False)


def test_meta_descriptor():
    assert MetaWithoutDocWithCopy.meta.__doc__ == ''
    assert MetaWithDocWithoutCopy.meta.__doc__ == 'test'

    a = MetaWithoutDocWithCopy()

    b = MetaWithDocWithoutCopy()

    # Check valid and invalid inputs
    for i in (a, b):

        assert isinstance(i.meta, OrderedDict)
        # after the first get access the _meta is set to be an Ordereddict
        assert isinstance(i._meta, OrderedDict)
        assert len(i.meta) == 0

        # Check that the private and public attribute are in sync
        i._meta['a'] = 100
        assert i.meta['a'] == 100
        i.meta['a'] = 0
        assert i.meta['a'] == 0

        # Setting to None sets the private attribute to an empty ordereddict.
        i.meta = None
        assert isinstance(i._meta, OrderedDict)
        assert isinstance(i.meta, OrderedDict)
        assert len(i.meta) == 0

        i.meta = {}
        assert isinstance(i.meta, dict)
        assert len(i.meta) == 0

        i.meta = Header()
        assert isinstance(i.meta, Header)
        assert len(i.meta) == 0


        with pytest.raises(TypeError):
            i.meta = 10

    meta = {'a': 10}
    a.meta = meta
    b.meta = meta
    meta['a'] = 5
    # a does copy so it has the original values
    assert a.meta['a'] == 10
    assert a.meta is not meta
    # b doesn't copy and only saves a reference
    assert b.meta['a'] == 5
    assert b.meta is meta
