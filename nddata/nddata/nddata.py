# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from .nddata_base import NDDataBase
from .mixins.ndarithmetic import NDArithmeticMixin
from .mixins.ndio import NDIOMixin
from .mixins.ndslicing import NDSlicingMixin
from .mixins.ndunitconversion import NDUnitConvMixin
from .mixins.ndstats import NDStatsMixin
from .mixins.ndclip import NDClippingMixin
from .mixins.ndreduce import NDReduceMixin
from .mixins.ndreshape import NDReshapeMixin
from .mixins.ndplot import NDPlottingMixin
from .mixins.ndfilter import NDFilterMixin


__all__ = ['NDData']


# The base class must be last in the bases!
class NDData(NDPlottingMixin,
             NDFilterMixin,
             NDReshapeMixin, NDSlicingMixin,
             NDUnitConvMixin, NDArithmeticMixin, NDClippingMixin,
             NDReduceMixin,
             NDIOMixin,
             NDStatsMixin, NDDataBase):
    """Implements `NDDataBase` with all Mixins.

    This class implements a `NDDataBase`-like container that supports reading
    and writing as implemented in the ``astropy.io.registry`` and also slicing
    (indexing) and simple arithmetics (add, subtract, divide and multiply).

    Examples
    --------
    The mixins allow operation that are not possible with `NDDataBase` or
    `~meta.NDDataMeta`, i.e. simple arithmetics::

        >>> from nddata.nddata import NDData, StdDevUncertainty
        >>> import numpy as np

        >>> data = np.ones((3,3), dtype=float)
        >>> ndd1 = NDData(data, uncertainty=StdDevUncertainty(data))
        >>> ndd2 = NDData(data, uncertainty=StdDevUncertainty(data))

        >>> ndd3 = ndd1.add(ndd2)
        >>> ndd3.data
        array([[ 2.,  2.,  2.],
               [ 2.,  2.,  2.],
               [ 2.,  2.,  2.]])
        >>> ndd3.uncertainty
        StdDevUncertainty([[ 1.41421356,  1.41421356,  1.41421356],
                           [ 1.41421356,  1.41421356,  1.41421356],
                           [ 1.41421356,  1.41421356,  1.41421356]])

    But also slicing (indexing) is possible::

        >>> ndd4 = ndd3[1,:]
        >>> ndd4.data
        array([ 2.,  2.,  2.])
        >>> ndd4.uncertainty
        StdDevUncertainty([ 1.41421356,  1.41421356,  1.41421356])

    See `~mixins.NDSlicingMixin` for a description how slicing works (which
    attributes) are sliced.
    """
    pass
