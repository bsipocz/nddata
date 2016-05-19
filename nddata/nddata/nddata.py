# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from .nddata_base import NDDataBase
from .mixins.ndarithmetic import NDArithmeticMixin
from .mixins.ndio import NDIOMixin
from .mixins.ndslicing import NDSlicingMixin


__all__ = ['NDData']


class NDData(NDArithmeticMixin, NDIOMixin, NDSlicingMixin, NDDataBase):
    """Implements `NDDataBase` with all Mixins.

    This class implements a `NDDataBase`-like container that supports reading
    and writing as implemented in the ``astropy.io.registry`` and also slicing
    (indexing) and simple arithmetics (add, subtract, divide and multiply).

    See also
    --------
    NDDataBase
    mixins.NDArithmeticMixin
    mixins.NDSlicingMixin
    mixins.NDIOMixin

    Examples
    --------
    The mixins allow operation that are not possible with `NDDataBase` or
    `~meta.NDDataMeta`, i.e. simple arithmetics::

        >>> from nddata.nddata import NDData, StdDevUncertainty
        >>> import numpy as np

        >>> data = np.ones((3,3), dtype=np.float)
        >>> ndd1 = NDData(data, uncertainty=StdDevUncertainty(data))
        >>> ndd2 = NDData(data, uncertainty=StdDevUncertainty(data))

        >>> ndd3 = ndd1.add(ndd2)
        >>> ndd3.data
        array([[ 2.,  2.,  2.],
               [ 2.,  2.,  2.],
               [ 2.,  2.,  2.]])
        >>> ndd3.uncertainty.array
        array([[ 1.41421356,  1.41421356,  1.41421356],
               [ 1.41421356,  1.41421356,  1.41421356],
               [ 1.41421356,  1.41421356,  1.41421356]])

    But also slicing (indexing) is possible::

        >>> ndd4 = ndd3[1,:]
        >>> ndd4.data
        array([ 2.,  2.,  2.])
        >>> ndd4.uncertainty.array
        array([ 1.41421356,  1.41421356,  1.41421356])

    See `~mixins.NDSlicingMixin` for a description how slicing works (which
    attributes) are sliced.
    """
    pass
