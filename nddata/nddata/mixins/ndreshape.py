# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from collections import Iterable

import numpy as np

from astropy import log
from astropy.wcs import WCS

from ...utils.copyutils import do_copy


__all__ = ['NDReshapeMixin']


class NDReshapeMixin(object):
    """Mixin to provide shape changing methods on classes implementing the \
            `~.NDDataBase` interface.

    Requires the ``uncertainty`` to have an ``offset`` method.

    ``mask``, ``flags``, ``wcs`` can be offsetted if they are `numpy.ndarray`.
    But the ``WCS`` also supports `~astropy.wcs.WCS` objects.

    The implementation follows the guideline of slicing that whenever possible
    the resulting attributes are references **and not copies** of the original
    attributes. This is not possible with `offset`.
    """
    def offset(self, pad_width):
        # Abort slicing if the data is a single scalar.
        if self.data is None or self.data.shape == ():
            raise TypeError('scalars and None data cannot be offsetted.')

        pad_width_iterable = isinstance(pad_width, Iterable)

        # To avoid special casing the "pad_width" in the different offsetting
        # methods - could go crazy with wcs or if some attribute doesn't have
        # the same dimensions as the data. Better to catch that here before
        # doing the special cases there. See np.lib.pad for a more extensive
        # explanation of the meanings.

        # The first case is if it's just an integer.
        if not pad_width_iterable:
            # This is interpreted as ((pad, pad_), ..., (pad, pad)) so we need
            # to create an iterable containing ndim (pad, pad) values:
            pad_width = [(pad_width, pad_width) for _ in range(self.data.ndim)]

        # The second case is when it's an iterable BUT the elements are no
        # iterables.
        elif pad_width_iterable and not isinstance(pad_width[0], Iterable):
            # This is interpreted as (pad, ... pad) so we expand it accordingly
            pad_width = [pad_width for _ in range(self.data.ndim)]

        # The last case is when it's an iterable of iterables, in that case we
        # don't need to do anything since it _should_ fail if the number of
        # dimensions doesn't match the one of the data.

        # Let the other methods handle slicing.
        kwargs = self._offset(pad_width)
        return self.__class__(**kwargs)

    def _offset(self, pad_width):
        """
        Collects the offsetted attributes and passes them back as `dict`.
        """
        kwargs = {}
        # Try to offset some attributes
        kwargs['data'] = self._offset_data(pad_width)
        kwargs['uncertainty'] = self._offset_uncertainty(pad_width)
        kwargs['mask'] = self._offset_mask(pad_width)
        kwargs['wcs'] = self._offset_wcs(pad_width)
        kwargs['flags'] = self._offset_flags(pad_width)
        # Attributes which are copied and not intended to be offsetted.
        kwargs['unit'] = self.unit
        kwargs['meta'] = do_copy(self.meta)
        return kwargs

    def _offset_data(self, pad_width):
        # Pad with zeros
        return np.lib.pad(self.data, pad_width, mode='constant',
                          constant_values=0)

    def _offset_uncertainty(self, pad_width):
        if self.uncertainty is None:
            return None
        try:
            return self.uncertainty.offset(pad_width)
        except AttributeError:
            # AttributeError in case it had no offset method.
            log.info("uncertainty cannot be offsetted.")
        return self.uncertainty

    def _offset_mask(self, pad_width):
        if self.mask is None:
            return None
        try:
            # Pad the mask. Interpret the offsetted elements as masked.
            return np.lib.pad(self.mask, pad_width, mode='constant',
                              constant_values=1)
        except ValueError:
            log.info("mask cannot be offsetted.")
        return self.mask

    def _offset_wcs(self, pad_width):
        if self.wcs is None:
            return None
        elif isinstance(self.wcs, WCS):
            wcs = self.wcs.deepcopy()
            # wcs naxis are reversed numpy axis, so we need to add the first
            # axis with the last "before"-padding
            wcs.wcs.crpix = [val + pad_width[-idx][0]
                             for idx, val in enumerate(wcs.wcs.crpix, 1)]
            return wcs
        else:
            # Try to interpret it as numpy ndarray?
            try:
                return np.lib.pad(self.flags, pad_width, mode='constant',
                                  constant_values=0)
            except ValueError:
                log.info("wcs cannot be offsetted.")
        return self.wcs

    def _offset_flags(self, pad_width):
        if self.flags is None:
            return None
        try:
            return np.lib.pad(self.flags, pad_width, mode='constant',
                              constant_values=0)
        except ValueError:
            log.info("flags cannot be offsetted.")
        return self.flags
