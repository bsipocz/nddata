# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from astropy.tests.helper import pytest

from ... import NDData


def test_extremaclip_axis_not_exist():
    ndd = NDData(np.array([5, 2, 1, 3, 3]))
    with pytest.raises(IndexError):
        ndd.clip_extrema(nlow=1, axis=1)


def test_extremaclip_no_clip():
    data = np.array([5, 2, 1, 3, 3])

    # no initial mask will stay None when clipped without nlow/nhigh
    ndd = NDData(data)
    ndd2 = ndd.copy()
    ndd2.clip_extrema()
    assert ndd2.mask is None

    # with initial mask it will also stay the same when clipped without
    # nlow/nhigh
    ndd = NDData(data, mask=(data > 2))
    ndd2 = ndd.copy()
    ndd2.clip_extrema()
    np.testing.assert_array_equal(ndd.mask, ndd2.mask)


def test_extremaclip_one_d_nlow():
    data = np.array([5, 2, 1, 3, 3])
    ndd = NDData(data)
    ndd.clip_extrema(nlow=1)
    np.testing.assert_array_equal(ndd.data, data)
    np.testing.assert_array_equal(ndd.mask,
                                  np.array([0, 0, 1, 0, 0], dtype=bool))

    ndd.clip_extrema(nlow=1)
    np.testing.assert_array_equal(ndd.data, data)
    np.testing.assert_array_equal(ndd.mask,
                                  np.array([0, 1, 1, 0, 0], dtype=bool))

    ndd.clip_extrema(nlow=1)
    np.testing.assert_array_equal(ndd.data, data)
    np.testing.assert_array_equal(ndd.mask,
                                  np.array([0, 1, 1, 1, 0], dtype=bool))

    ndd.clip_extrema(nlow=1)
    np.testing.assert_array_equal(ndd.data, data)
    np.testing.assert_array_equal(ndd.mask,
                                  np.array([0, 1, 1, 1, 1], dtype=bool))

    ndd.clip_extrema(nlow=1)
    np.testing.assert_array_equal(ndd.data, data)
    np.testing.assert_array_equal(ndd.mask,
                                  np.array([1, 1, 1, 1, 1], dtype=bool))

    ndd.clip_extrema(nlow=1)
    np.testing.assert_array_equal(ndd.data, data)
    np.testing.assert_array_equal(ndd.mask,
                                  np.array([1, 1, 1, 1, 1], dtype=bool))


def test_extremaclip_one_d_nlow():
    data = np.array([5, 2, 1, 3, 3])
    ndd = NDData(data)
    ndd.clip_extrema(nlow=1, axis=0)
    np.testing.assert_array_equal(ndd.data, data)
    np.testing.assert_array_equal(ndd.mask,
                                  np.array([0, 0, 1, 0, 0], dtype=bool))

    ndd.clip_extrema(nlow=1, axis=0)
    np.testing.assert_array_equal(ndd.data, data)
    np.testing.assert_array_equal(ndd.mask,
                                  np.array([0, 1, 1, 0, 0], dtype=bool))

    ndd.clip_extrema(nlow=1, axis=0)
    np.testing.assert_array_equal(ndd.data, data)
    np.testing.assert_array_equal(ndd.mask,
                                  np.array([0, 1, 1, 1, 0], dtype=bool))

    ndd.clip_extrema(nlow=1, axis=0)
    np.testing.assert_array_equal(ndd.data, data)
    np.testing.assert_array_equal(ndd.mask,
                                  np.array([0, 1, 1, 1, 1], dtype=bool))

    ndd.clip_extrema(nlow=1, axis=0)
    np.testing.assert_array_equal(ndd.data, data)
    np.testing.assert_array_equal(ndd.mask,
                                  np.array([1, 1, 1, 1, 1], dtype=bool))

    ndd.clip_extrema(nlow=1, axis=0)
    np.testing.assert_array_equal(ndd.data, data)
    np.testing.assert_array_equal(ndd.mask,
                                  np.array([1, 1, 1, 1, 1], dtype=bool))


def test_extremaclip_one_d_nhigh():
    data = np.array([5, 2, 1, 3, 3])
    ndd = NDData(data)
    ndd.clip_extrema(nhigh=1)
    np.testing.assert_array_equal(ndd.data, data)
    np.testing.assert_array_equal(ndd.mask,
                                  np.array([1, 0, 0, 0, 0], dtype=bool))

    ndd.clip_extrema(nhigh=1)
    np.testing.assert_array_equal(ndd.data, data)
    np.testing.assert_array_equal(ndd.mask,
                                  np.array([1, 0, 0, 1, 0], dtype=bool))

    ndd.clip_extrema(nhigh=1)
    np.testing.assert_array_equal(ndd.data, data)
    np.testing.assert_array_equal(ndd.mask,
                                  np.array([1, 0, 0, 1, 1], dtype=bool))

    ndd.clip_extrema(nhigh=1)
    np.testing.assert_array_equal(ndd.data, data)
    np.testing.assert_array_equal(ndd.mask,
                                  np.array([1, 1, 0, 1, 1], dtype=bool))

    ndd.clip_extrema(nhigh=1)
    np.testing.assert_array_equal(ndd.data, data)
    np.testing.assert_array_equal(ndd.mask,
                                  np.array([1, 1, 1, 1, 1], dtype=bool))

    ndd.clip_extrema(nhigh=1)
    np.testing.assert_array_equal(ndd.data, data)
    np.testing.assert_array_equal(ndd.mask,
                                  np.array([1, 1, 1, 1, 1], dtype=bool))


def test_extremaclip_one_d_nlow_nhigh():
    data = np.array([5, 2, 1, 3, 3])
    ndd = NDData(data)
    ndd.clip_extrema(nlow=1, nhigh=1)
    np.testing.assert_array_equal(ndd.data, data)
    np.testing.assert_array_equal(ndd.mask,
                                  np.array([1, 0, 1, 0, 0], dtype=bool))

    ndd.clip_extrema(nlow=1, nhigh=1)
    np.testing.assert_array_equal(ndd.data, data)
    np.testing.assert_array_equal(ndd.mask,
                                  np.array([1, 1, 1, 1, 0], dtype=bool))

    ndd.clip_extrema(nlow=1, nhigh=1)
    np.testing.assert_array_equal(ndd.data, data)
    np.testing.assert_array_equal(ndd.mask,
                                  np.array([1, 1, 1, 1, 1], dtype=bool))


def test_extremaclip_one_d_nlow_nhigh_higher():
    data = np.array([5, 2, 1, 3, 3])
    ndd = NDData(data)
    ndd.clip_extrema(nlow=2, nhigh=2)
    np.testing.assert_array_equal(ndd.data, data)
    np.testing.assert_array_equal(ndd.mask,
                                  np.array([1, 1, 1, 1, 0], dtype=bool))


def test_extremaclip_multi_d_nlow_no_axis():
    data = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    ndd = NDData(data)
    ndd.clip_extrema(nlow=1, axis=None)
    mask = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=bool)
    np.testing.assert_array_equal(ndd.data, data)
    np.testing.assert_array_equal(ndd.mask, mask)

    ndd.clip_extrema(nlow=1, axis=None)
    mask = np.array([[1, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=bool)
    np.testing.assert_array_equal(ndd.data, data)
    np.testing.assert_array_equal(ndd.mask, mask)

    ndd.clip_extrema(nlow=1, axis=None)
    mask = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]], dtype=bool)
    np.testing.assert_array_equal(ndd.data, data)
    np.testing.assert_array_equal(ndd.mask, mask)


def test_extremaclip_multi_d_nlow_first_axis():
    data = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    ndd = NDData(data)
    ndd.clip_extrema(nlow=1, axis=0)
    mask = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]], dtype=bool)
    np.testing.assert_array_equal(ndd.data, data)
    np.testing.assert_array_equal(ndd.mask, mask)

    ndd.clip_extrema(nlow=1, axis=0)
    mask = np.array([[1, 1, 1], [1, 1, 1], [0, 0, 0]], dtype=bool)
    np.testing.assert_array_equal(ndd.data, data)
    np.testing.assert_array_equal(ndd.mask, mask)

    ndd.clip_extrema(nlow=2, axis=0)
    mask = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=bool)
    np.testing.assert_array_equal(ndd.data, data)
    np.testing.assert_array_equal(ndd.mask, mask)


def test_extremaclip_multi_d_nhigh_first_axis():
    data = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    ndd = NDData(data)
    ndd.clip_extrema(nhigh=1, axis=0)
    mask = np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1]], dtype=bool)
    np.testing.assert_array_equal(ndd.data, data)
    np.testing.assert_array_equal(ndd.mask, mask)

    ndd.clip_extrema(nhigh=1, axis=0)
    mask = np.array([[0, 0, 0], [1, 1, 1], [1, 1, 1]], dtype=bool)
    np.testing.assert_array_equal(ndd.data, data)
    np.testing.assert_array_equal(ndd.mask, mask)

    ndd.clip_extrema(nhigh=2, axis=0)
    mask = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=bool)
    np.testing.assert_array_equal(ndd.data, data)
    np.testing.assert_array_equal(ndd.mask, mask)


def test_extremaclip_multi_d_nlow_second_axis():
    data = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    ndd = NDData(data)
    ndd.clip_extrema(nlow=1, axis=1)
    mask = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=bool)
    np.testing.assert_array_equal(ndd.data, data)
    np.testing.assert_array_equal(ndd.mask, mask)

    ndd.clip_extrema(nlow=1, axis=1)
    mask = np.array([[1, 1, 0], [1, 1, 0], [1, 1, 0]], dtype=bool)
    np.testing.assert_array_equal(ndd.data, data)
    np.testing.assert_array_equal(ndd.mask, mask)

    ndd.clip_extrema(nlow=2, axis=1)
    mask = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=bool)
    np.testing.assert_array_equal(ndd.data, data)
    np.testing.assert_array_equal(ndd.mask, mask)


def test_extremaclip_multi_d_nhigh_second_axis():
    data = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    ndd = NDData(data)
    ndd.clip_extrema(nhigh=1, axis=1)
    mask = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=bool)
    np.testing.assert_array_equal(ndd.data, data)
    np.testing.assert_array_equal(ndd.mask, mask)

    ndd.clip_extrema(nhigh=1, axis=1)
    mask = np.array([[0, 1, 1], [0, 1, 1], [0, 1, 1]], dtype=bool)
    np.testing.assert_array_equal(ndd.data, data)
    np.testing.assert_array_equal(ndd.mask, mask)

    ndd.clip_extrema(nhigh=2, axis=1)
    mask = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=bool)
    np.testing.assert_array_equal(ndd.data, data)
    np.testing.assert_array_equal(ndd.mask, mask)


def test_extremaclip_3d():
    data = np.array([[[2, 8, 9],
                      [1, 6, 2],
                      [9, 9, 8]],

                     [[8, 2, 5],
                      [3, 2, 0],
                      [3, 8, 0]],

                     [[5, 3, 7],
                      [3, 6, 7],
                      [3, 5, 1]]])

    ndd = NDData(data)

    ndd.clip_extrema(nlow=1, axis=0)
    ref_mask = np.array([[[ True, False, False],
                          [ True, False, False],
                          [False, False, False]],

                         [[False,  True,  True],
                          [False,  True,  True],
                          [ True, False,  True]],

                         [[False, False, False],
                          [False, False, False],
                          [False,  True, False]]], dtype=bool)
    np.testing.assert_array_equal(ndd.mask, ref_mask)

    ndd.clip_extrema(nhigh=1, axis=1)
    ref_mask = np.array([[[ True, False,  True],
                          [ True, False, False],
                          [ True,  True, False]],

                         [[ True,  True,  True],
                          [False,  True,  True],
                          [ True,  True,  True]],

                         [[ True, False,  True],
                          [False,  True, False],
                          [False,  True, False]]], dtype=bool)
    np.testing.assert_array_equal(ndd.mask, ref_mask)

    ndd.clip_extrema(nlow=1, axis=2)
    ref_mask = np.array([[[ True,  True,  True],
                          [ True, False,  True],
                          [ True,  True,  True]],

                         [[ True,  True,  True],
                          [ True,  True,  True],
                          [ True,  True,  True]],

                         [[ True,  True,  True],
                          [ True,  True, False],
                          [False,  True,  True]]], dtype=bool)
    np.testing.assert_array_equal(ndd.mask, ref_mask)

