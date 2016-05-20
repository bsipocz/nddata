# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from astropy.tests.helper import pytest

from ..nduncertainty_stddev import StdDevUncertainty
from ..nduncertainty_unknown import UnknownUncertainty
from ..meta import NDUncertainty
from ..exceptions import IncompatibleUncertaintiesException
from ..exceptions import MissingDataAssociationException
from ..nddata import NDDataBase


def test_unknown_to_stddev():
    # This conversion does:
    # - not copy the data
    # - keep the values/unit/parent_nddata
    unc1 = UnknownUncertainty(np.ones(5), unit='m')
    unc2 = StdDevUncertainty.from_uncertainty(unc1)
    assert unc1.data is unc2.data
    assert unc1.unit == unc2.unit
    with pytest.raises(MissingDataAssociationException):
        unc1.parent_nddata
    with pytest.raises(MissingDataAssociationException):
        unc2.parent_nddata

    # with parent
    parent = NDDataBase(1)
    parent.uncertainty = unc1

    unc3 = StdDevUncertainty.from_uncertainty(unc1)
    assert unc3.data is unc1.data
    assert unc3.unit == unc1.unit
    assert unc1.parent_nddata is unc3.parent_nddata
    assert unc1.parent_nddata is parent

    # make sure unc2 wasn't modified ...
    with pytest.raises(MissingDataAssociationException):
        unc2.parent_nddata

    # Less efficient direct usage during init
    unc4 = StdDevUncertainty(unc1)
    assert unc4.data is unc1.data
    assert unc4.unit == unc1.unit
    assert unc1.parent_nddata is unc4.parent_nddata
    assert unc1.parent_nddata is parent


def test_stddev_to_unknown():
    # This conversion does:
    # - not copy the data
    # - keep the values/unit/parent_nddata
    unc1 = StdDevUncertainty(np.ones(5), unit='m')
    unc2 = UnknownUncertainty.from_uncertainty(unc1)
    assert unc1.data is unc2.data
    assert unc1.unit == unc2.unit
    with pytest.raises(MissingDataAssociationException):
        unc1.parent_nddata
    with pytest.raises(MissingDataAssociationException):
        unc2.parent_nddata

    # with parent
    parent = NDDataBase(1)
    parent.uncertainty = unc1

    unc3 = UnknownUncertainty.from_uncertainty(unc1)
    assert unc3.data is unc1.data
    assert unc3.unit == unc1.unit
    assert unc1.parent_nddata is unc3.parent_nddata
    assert unc1.parent_nddata is parent

    # make sure unc2 wasn't modified ...
    with pytest.raises(MissingDataAssociationException):
        unc2.parent_nddata

    # Less efficient direct usage during init
    unc4 = UnknownUncertainty(unc1)
    assert unc4.data is unc1.data
    assert unc4.unit == unc1.unit
    assert unc1.parent_nddata is unc4.parent_nddata
    assert unc1.parent_nddata is parent
