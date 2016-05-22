# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from .ndarithmetic import NDArithmeticMixin


__all__ = ['NDArithmeticPyOpsMixin']


class NDArithmeticPyOpsMixin(NDArithmeticMixin):
    """This mixin builds on `NDArithmeticMixin` by allowing also the use of \
            Python operators like ``+``, ``-``, ...

    .. warning::
        If you are using numpy 1.9 or earlier you should avoid any calculations
        where the first operand is a `numpy.ma.MaskedArray`.

    .. warning::
        You cannot use this mixin together with `~nddata.nddata.NDData`. You
        need to create a subclass of `~nddata.nddata.NDDataBase`!

    Examples
    --------
    To create a class with this mixin you must create a subclass yourself. The
    reason why the arithmetic operators are not allowed is because there are a
    lot of optional arguments regulating how arithmetic is handled on a
    `~nddata.nddata.NDDataBase`-object and forcing you (the user) to use the
    methods gives you complete control over the arithmetic operation while
    ``+`` simply hides what options there are::

        >>> from nddata.nddata import NDDataBase
        >>> from nddata.nddata.mixins import NDArithmeticPyOpsMixin
        >>> class NDData2(NDArithmeticPyOpsMixin, NDDataBase):
        ...     pass

        >>> ndd = NDData2(100, unit='m')

        >>> ndd + ndd
        NDData2(200.0)

        >>> 5 * ndd
        NDData2(500.0)

    The default parameters are used see
    `~nddata.nddata.ContextArithmeticDefaults` if you want to change them
    temporarly or permanently.
    """

    # To avoid that np.ndarrays mess up the result if they are the first
    # argument this should be higher than any other np.ndarray including
    # MaskedArrays and Quantities.
    __array_priority__ = 10000000

    def __add__(self, other):
        return self.add(other)

    def __radd__(self, other):
        return self.add(other, self)

    def __sub__(self, other):
        return self.subtract(other)

    def __rsub__(self, other):
        return self.subtract(other, self)

    def __mul__(self, other):
        return self.multiply(other)

    def __rmul__(self, other):
        return self.multiply(other, self)

    def __truediv__(self, other):
        return self.divide(other)

    def __rtruediv__(self, other):
        return self.divide(other, self)

    # The __div__ and __rdiv__ shouldn't be necessary since I imported
    # __future__.division but I don't know in which contexts someone might
    # happen to get into those (maybe with operator.__div__?!) so better leave
    # them in here.
    def __div__(self, other):  # pragma: no cover
        return self.divide(other)

    def __rdiv__(self, other):  # pragma: no cover
        return self.divide(other, self)

    def __pow__(self, other):
        return self.power(other)

    def __rpow__(self, other):
        return self.power(other, self)

    def __neg__(self):
        res = self._copy_without_data()
        res.data = self.data * -1
        return res

    def __pos__(self):
        return self.copy()

    def __abs__(self):
        res = self._copy_without_data()
        res.data = np.abs(self.data)
        return res
