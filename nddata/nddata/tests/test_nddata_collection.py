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
