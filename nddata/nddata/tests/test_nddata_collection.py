# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from astropy.tests.helper import pytest
from .. import NDDataCollection, NDData, NDDataBase
from .. import VarianceUncertainty, StdDevUncertainty


def test_collection_fails():
    # Class mismatch
    ndd1 = NDData(np.ones(3) * 1)
    ndd2 = NDDataBase(np.ones(3) * 2)

    ndds = NDDataCollection(ndd1, ndd2)
    with pytest.raises(ValueError):
        ndds.stack()

    ndds = NDDataCollection(ndd2, ndd1)
    with pytest.raises(ValueError):
        ndds.stack()

    # unit mismatch
    ndd1 = NDDataBase(np.ones(3) * 2, unit='m')
    ndd2 = NDDataBase(np.ones(3) * 2, unit='cm')

    ndds = NDDataCollection(ndd1, ndd2)
    with pytest.raises(ValueError):
        ndds.stack()

    ndds = NDDataCollection(ndd2, ndd1)
    with pytest.raises(ValueError):
        ndds.stack()

    # shape mismatch (data)
    ndd1 = NDDataBase(np.ones(4) * 2)
    ndd2 = NDDataBase(np.ones(3) * 2)

    ndds = NDDataCollection(ndd1, ndd2)
    with pytest.raises(ValueError):
        ndds.stack()

    ndds = NDDataCollection(ndd2, ndd1)
    with pytest.raises(ValueError):
        ndds.stack()

    # shape mismatch (mask to data)
    ndd1 = NDDataBase(np.ones(3) * 2, mask=np.ones(3, dtype=bool))
    ndd2 = NDDataBase(np.ones(3) * 2, mask=np.ones(4, dtype=bool))

    ndds = NDDataCollection(ndd1, ndd2)
    with pytest.raises(ValueError):
        ndds.stack()

    ndds = NDDataCollection(ndd2, ndd1)
    with pytest.raises(ValueError):
        ndds.stack()

    # shape mismatch (flags to data)
    ndd1 = NDDataBase(np.ones(3) * 2, flags=np.ones(3, dtype=bool))
    ndd2 = NDDataBase(np.ones(3) * 2, flags=np.ones(4, dtype=bool))

    ndds = NDDataCollection(ndd1, ndd2)
    with pytest.raises(ValueError):
        ndds.stack()

    ndds = NDDataCollection(ndd2, ndd1)
    with pytest.raises(ValueError):
        ndds.stack()

    # shape mismatch (uncertainty to data)
    ndd1 = NDDataBase(np.ones(3) * 2,
                      uncertainty=VarianceUncertainty(np.ones(3)))
    ndd2 = NDDataBase(np.ones(3) * 2,
                      uncertainty=VarianceUncertainty(np.ones(4)))

    ndds = NDDataCollection(ndd1, ndd2)
    with pytest.raises(ValueError):
        ndds.stack()

    ndds = NDDataCollection(ndd2, ndd1)
    with pytest.raises(ValueError):
        ndds.stack()

    # uncertainty class mismatch
    ndd1 = NDDataBase(np.ones(3) * 2,
                      uncertainty=VarianceUncertainty(np.ones(3)))
    ndd2 = NDDataBase(np.ones(3) * 2,
                      uncertainty=StdDevUncertainty(np.ones(3)))

    ndds = NDDataCollection(ndd1, ndd2)
    with pytest.raises(ValueError):
        ndds.stack()

    ndds = NDDataCollection(ndd2, ndd1)
    with pytest.raises(ValueError):
        ndds.stack()

    # uncertainty unit mismatch
    ndd1 = NDDataBase(np.ones(3) * 2,
                      uncertainty=StdDevUncertainty(np.ones(3)))
    ndd2 = NDDataBase(np.ones(3) * 2,
                      uncertainty=StdDevUncertainty(np.ones(3), unit='m'))

    ndds = NDDataCollection(ndd1, ndd2)
    with pytest.raises(ValueError):
        ndds.stack()

    ndds = NDDataCollection(ndd2, ndd1)
    with pytest.raises(ValueError):
        ndds.stack()


def test_collection_uncertainty():
    ndd1 = NDDataBase(np.ones(3) * 2)
    ndd2 = NDDataBase(np.ones(3) * 3,
                      uncertainty=StdDevUncertainty(np.ones(3), unit='m'))
    ndds = NDDataCollection(ndd1, ndd2)
    stack = ndds.stack()

    np.testing.assert_array_equal(stack.data, [[2, 2, 2], [3, 3, 3]])
    np.testing.assert_array_equal(stack.uncertainty.data,
                                  [[0, 0, 0], [1, 1, 1]])
    assert isinstance(stack.uncertainty, StdDevUncertainty)
    assert stack.uncertainty.unit == 'm'

    ndds = NDDataCollection(ndd2, ndd1)
    stack = ndds.stack()

    np.testing.assert_array_equal(stack.data, [[3, 3, 3], [2, 2, 2]])
    np.testing.assert_array_equal(stack.uncertainty.data,
                                  [[1, 1, 1], [0, 0, 0]])
    assert isinstance(stack.uncertainty, StdDevUncertainty)
    assert stack.uncertainty.unit == 'm'

    # Test with 3:
    ndd3 = NDDataBase(np.ones(3) * 1,
                      uncertainty=StdDevUncertainty(np.ones(3) * 2, unit='m'))
    ndds = NDDataCollection(ndd3, ndd1, ndd2)
    stack = ndds.stack()

    np.testing.assert_array_equal(stack.data,
                                  [[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    np.testing.assert_array_equal(stack.uncertainty.data,
                                  [[2, 2, 2], [0, 0, 0], [1, 1, 1]])
    assert isinstance(stack.uncertainty, StdDevUncertainty)
    assert stack.uncertainty.unit == 'm'


def test_collection_mask():
    ndd1 = NDDataBase(np.ones(3) * 2)
    ndd2 = NDDataBase(np.ones(3) * 3, mask=np.ones(3, dtype=bool))
    ndds = NDDataCollection(ndd1, ndd2)
    stack = ndds.stack()

    np.testing.assert_array_equal(stack.data, [[2, 2, 2], [3, 3, 3]])
    np.testing.assert_array_equal(stack.mask, [[0, 0, 0], [1, 1, 1]])

    ndds = NDDataCollection(ndd2, ndd1)
    stack = ndds.stack()

    np.testing.assert_array_equal(stack.data, [[3, 3, 3], [2, 2, 2]])
    np.testing.assert_array_equal(stack.mask, [[1, 1, 1], [0, 0, 0]])

    # Test with 3:
    ndd3 = NDDataBase(np.ones(3) * 1, mask=np.ones(3, dtype=bool))
    ndds = NDDataCollection(ndd3, ndd1, ndd2)
    stack = ndds.stack()

    np.testing.assert_array_equal(stack.data,
                                  [[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    np.testing.assert_array_equal(stack.mask,
                                  [[1, 1, 1], [0, 0, 0], [1, 1, 1]])


def test_collection_flags():
    ndd1 = NDDataBase(np.ones(3) * 2)
    ndd2 = NDDataBase(np.ones(3) * 3, flags=np.ones(3, dtype=int))
    ndds = NDDataCollection(ndd1, ndd2)
    stack = ndds.stack()

    np.testing.assert_array_equal(stack.data, [[2, 2, 2], [3, 3, 3]])
    np.testing.assert_array_equal(stack.flags, [[0, 0, 0], [1, 1, 1]])

    ndds = NDDataCollection(ndd2, ndd1)
    stack = ndds.stack()

    np.testing.assert_array_equal(stack.data, [[3, 3, 3], [2, 2, 2]])
    np.testing.assert_array_equal(stack.flags, [[1, 1, 1], [0, 0, 0]])

    # Test with 3:
    ndd3 = NDDataBase(np.ones(3) * 1, flags=np.ones(3, dtype=int)*2)
    ndds = NDDataCollection(ndd3, ndd1, ndd2)
    stack = ndds.stack()

    np.testing.assert_array_equal(stack.data,
                                  [[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    np.testing.assert_array_equal(stack.flags,
                                  [[2, 2, 2], [0, 0, 0], [1, 1, 1]])


def test_collection_meta_unit_wcs():
    ndd1 = NDDataBase(np.ones(3), meta={'a': 1}, unit='m', wcs=1)
    ndd2 = NDDataBase(np.ones(3), meta={'a': 2}, unit='m', wcs=2)
    ndd3 = NDDataBase(np.ones(3), meta={'a': 3}, unit='m', wcs=3)

    ndds = NDDataCollection(ndd1, ndd2, ndd3)
    stack = ndds.stack()
    assert stack.meta['a'] == 1
    assert stack.unit == 'm'
    assert stack.wcs == 1

    ndds = NDDataCollection(ndd3, ndd1, ndd2)
    stack = ndds.stack()
    assert stack.meta['a'] == 3
    assert stack.unit == 'm'
    assert stack.wcs == 3

    ndds = NDDataCollection(ndd2, ndd3, ndd1)
    stack = ndds.stack()
    assert stack.meta['a'] == 2
    assert stack.unit == 'm'
    assert stack.wcs == 2


def test_collection_allmeta_func():
    ndd1 = NDDataBase(3, meta={'a': 1})
    ndd2 = NDDataBase(3, meta={'a': 2})
    ndd3 = NDDataBase(3, meta={'a': 3})
    ndds = NDDataCollection(ndd1, ndd2, ndd3)
    # A stupid function to convert it to NDData but well ... it works :-/
    # TODO: Use a good function here!!!
    np.testing.assert_array_equal(ndds.get_all_metas(func=NDData)['a'],
                                  [1, 2, 3])
