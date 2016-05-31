# Licensed under a 3-clause BSD style license - see LICENSE.rst

# TEST_UNICODE_LITERALS

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from numpy.testing import assert_array_equal

from astropy.tests.helper import pytest
from astropy import units as u
from astropy import wcs

from ... import NDData, StdDevUncertainty


def test_pad_fail():
    with pytest.raises(TypeError):
        NDData(None).offset(10)
    with pytest.raises(TypeError):
        NDData(10).offset(1)


def test_pad_width_int():
    kwargs = {'data': np.ones((3, 3)),
              'mask': np.zeros((3, 3), dtype=bool),
              'flags': np.ones((3, 3)),
              'wcs': np.arange(1, 10).reshape(3, 3),
              'meta': {'a': 1},
              'unit': 'm',
              }

    ndd = NDData(**kwargs)

    # Integer is interpreted as ((3, 3), (3, 3))
    nddoffset = ndd.offset(3)

    # Assert right final shape
    assert nddoffset.data.shape == (9, 9)
    assert nddoffset.mask.shape == (9, 9)
    assert nddoffset.wcs.shape == (9, 9)
    assert nddoffset.flags.shape == (9, 9)

    # Assert right fill value
    assert nddoffset.data[0, 0] == 0
    assert nddoffset.mask[0, 0] == 1
    assert nddoffset.wcs[0, 0] == 0
    assert nddoffset.flags[0, 0] == 0

    # Assert meta is not a copy
    ndd.meta['a'] = 10
    assert nddoffset.meta['a'] == 1

    # Assert unit stayed
    assert nddoffset.unit == u.m


def test_pad_width_int_data_only1d():
    ndd = NDData(np.ones(3))

    # Integer is interpreted as ((3, 3),)
    nddoffset = ndd.offset(3)

    # Assert right final shape
    assert nddoffset.data.shape == (9, )

    # Assert right fill value
    assert nddoffset.data[0] == 0

    assert not nddoffset.meta
    assert nddoffset.unit is None
    assert nddoffset.mask is None
    assert nddoffset.uncertainty is None
    assert nddoffset.wcs is None
    assert nddoffset.flags is None


def test_pad_width_int_data_only2d():
    ndd = NDData(np.ones((3, 3)))

    # Integer is interpreted as ((3, 3), (3, 3))
    nddoffset = ndd.offset(3)

    # Assert right final shape
    assert nddoffset.data.shape == (9, 9)

    # Assert right fill value
    assert nddoffset.data[0, 0] == 0

    assert not nddoffset.meta
    assert nddoffset.unit is None
    assert nddoffset.mask is None
    assert nddoffset.uncertainty is None
    assert nddoffset.wcs is None
    assert nddoffset.flags is None


def test_pad_width_int_unoffsettables():
    kwargs = {'data': np.ones((3, 3)),
              'mask': 5,
              'flags': 5,
              'wcs': 5,
              'uncertainty': StdDevUncertainty(5),
              'meta': {'a': 1},
              'unit': 'm',
              }

    ndd = NDData(**kwargs)

    # Integer is interpreted as ((3, 3), (3, 3))
    nddoffset = ndd.offset(3)

    # Assert right final shape
    assert nddoffset.data.shape == (9, 9)
    # The others were tried to be offsetted but cannot, so they stay the same.
    assert nddoffset.mask == 5
    assert nddoffset.wcs == 5
    assert nddoffset.flags == 5

    # Assert right fill value
    assert nddoffset.data[0, 0] == 0


def test_pad_width_tuple():
    kwargs = {'data': np.ones((3, 3)),
              'mask': np.zeros((3, 3), dtype=bool),
              'flags': np.ones((3, 3)),
              'wcs': np.arange(1, 10).reshape(3, 3),
              'meta': {'a': 1},
              'unit': 'm',
              }

    ndd = NDData(**kwargs)

    # single tuple is interpreted as ((3, 3), (3, 3))
    nddoffset = ndd.offset((3, 3))

    # Assert right final shape
    assert nddoffset.data.shape == (9, 9)
    assert nddoffset.mask.shape == (9, 9)
    assert nddoffset.wcs.shape == (9, 9)
    assert nddoffset.flags.shape == (9, 9)

    # Assert right fill value
    assert nddoffset.data[0, 0] == 0
    assert nddoffset.mask[0, 0] == 1
    assert nddoffset.wcs[0, 0] == 0
    assert nddoffset.flags[0, 0] == 0

    # Assert meta is not a copy
    ndd.meta['a'] = 10
    assert nddoffset.meta['a'] == 1

    # Assert unit stayed
    assert nddoffset.unit == u.m


def test_pad_width_tuple_tuple():
    kwargs = {'data': np.ones((3, 3)),
              'mask': np.zeros((3, 3), dtype=bool),
              'flags': np.ones((3, 3)),
              'wcs': np.arange(1, 10).reshape(3, 3),
              'meta': {'a': 1},
              'unit': 'm',
              }

    ndd = NDData(**kwargs)

    # iterable of iterable is interpreted as is.
    nddoffset = ndd.offset([(3, 3), (3, 3)])

    # Assert right final shape
    assert nddoffset.data.shape == (9, 9)
    assert nddoffset.mask.shape == (9, 9)
    assert nddoffset.wcs.shape == (9, 9)
    assert nddoffset.flags.shape == (9, 9)

    # Assert right fill value
    assert nddoffset.data[0, 0] == 0
    assert nddoffset.mask[0, 0] == 1
    assert nddoffset.wcs[0, 0] == 0
    assert nddoffset.flags[0, 0] == 0

    # Assert meta is not a copy
    ndd.meta['a'] = 10
    assert nddoffset.meta['a'] == 1

    # Assert unit stayed
    assert nddoffset.unit == u.m


def test_pad_wcs():

    w = wcs.WCS(naxis=2)

    w.wcs.crpix = [-234.75, 8.3393]
    w.wcs.cdelt = np.array([-0.066667, 0.066667])
    w.wcs.crval = [0, -90]
    w.wcs.ctype = ["RA---AIR", "DEC--AIR"]
    w.wcs.set_pv([(2, 1, 45.0)])

    ndd = NDData(np.ones((3, 3)), wcs=w)

    # Integer
    nddoffset = ndd.offset(3)
    assert_array_equal(nddoffset.wcs.wcs.crpix, [-231.75, 11.3393])

    # Tuple
    nddoffset = ndd.offset((3, 3))
    assert_array_equal(nddoffset.wcs.wcs.crpix, [-231.75, 11.3393])

    # List of tuple
    nddoffset = ndd.offset([(3, 3), (3, 3)])
    assert_array_equal(nddoffset.wcs.wcs.crpix, [-231.75, 11.3393])


def test_pad_uncertainty():

    uncertainty = StdDevUncertainty(np.ones((3, 3)))

    ndd = NDData(np.ones((3, 3)), uncertainty=uncertainty)

    # Integer
    nddoffset = ndd.offset(3)
    assert nddoffset.uncertainty.data.shape == (9, 9)
    assert nddoffset.uncertainty.data[0, 0] == 0

    # Tuple
    nddoffset = ndd.offset((3, 3))
    assert nddoffset.uncertainty.data.shape == (9, 9)
    assert nddoffset.uncertainty.data[0, 0] == 0

    # List of tuple
    nddoffset = ndd.offset([(3, 3), (3, 3)])
    assert nddoffset.uncertainty.data.shape == (9, 9)
    assert nddoffset.uncertainty.data[0, 0] == 0
