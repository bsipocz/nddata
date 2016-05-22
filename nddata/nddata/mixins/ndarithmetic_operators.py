# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from .ndarithmetic import NDArithmeticMixin


__all__ = ['NDArithmeticMixinPyOps']


class NDArithmeticMixinPyOps(NDArithmeticMixin):
    """This mixin builds on `NDArithmeticMixin` by allowing also the use of \
            Python operators like ``+``, ``-``, ...
    """
    defaults = {'propagate_uncertainties': True,
                'handle_mask': np.logical_or,
                'handle_meta': None,
                'handle_flags': None,
                'compare_wcs': 'first_found',
                'uncertainty_correlation': 0}

    # To avoid that np.ndarrays mess up the result if they are the first
    # argument this should be higher than any other np.ndarray including
    # MaskedArrays and Quantities.
    __array_priority__ = 10000000

    def __add__(self, other):
        return self.add(other, **self.defaults)

    def __radd__(self, other):
        return self.add(other, self, **self.defaults)

    def __sub__(self, other):
        return self.subtract(other, **self.defaults)

    def __rsub__(self, other):
        return self.subtract(other, self, **self.defaults)

    def __mul__(self, other):
        return self.multiply(other, **self.defaults)

    def __rmul__(self, other):
        return self.multiply(other, self, **self.defaults)

    def __div__(self, other):
        return self.divide(other, **self.defaults)

    def __rdiv__(self, other):
        return self.divide(other, self, **self.defaults)

    def __truediv__(self, other):
        return self.divide(other, **self.defaults)

    def __rtruediv__(self, other):
        return self.divide(other, self, **self.defaults)

    def __pow__(self, other):
        return self.power(other, **self.defaults)

    def __rpow__(self, other):
        return self.power(other, self, **self.defaults)

    def __neg__(self):
        res = self.copy()
        # TODO: This will copy the data twice...
        res.data = res.data * -1
        return res

    def __pos__(self):
        return self.copy()

    def __abs__(self):
        res = self.copy()
        # TODO: This will copy the data twice...
        res.data = np.abs(res.data)
        return res
