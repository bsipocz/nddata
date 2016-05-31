# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from collections import Iterable

from astropy import log
from astropy.wcs import WCS

from ...utils.copyutils import do_copy
from ...utils.numpyutils import pad


__all__ = ['NDReshapeMixin']


class NDReshapeMixin(object):
    """Mixin to provide shape changing methods on classes implementing the \
            `~.NDDataBase` interface.
    """
    def offset(self, pad_width):
        """Offset the instance by some padding width.

        See also
        --------
        numpy.pad
        nddata.utils.numpyutils.pad

        Parameters
        ----------
        pad_width : `int`, `tuple` or `tuple` of `tuple`
            If it is an integer it is interpreted as padding width in all
            dimensions. If it is a tuple it's first element is interpreted as
            padding in front of all dimensions and the second element as
            padding width at the end of all dimensions. If it's a tuple of
            tuples it must contain as many tuples as the ``data`` has
            dimensions and each tuple contain 2 values. This is roughly
            equivalent to the ``pad_width`` parameter of :func:`numpy.pad`.

        Returns
        -------
        offset_ndd : same class as instance
            The offsetted `~nddata.nddata.NDDataBase`-instance.

        Examples
        --------
        For example to pad with 1 elements along all dimensions::

            >>> from nddata.nddata import NDData
            >>> import numpy as np
            >>> ndd = NDData(np.ones((3, 3), int), mask=np.zeros((3,3), bool))

            >>> nddo = ndd.offset(1)
            >>> nddo
            NDData([[0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0],
                    [0, 1, 1, 1, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0]])

        While the ``data``, ``uncertainty``, ``flags`` and ``wcs`` are padded
        with zeros - if they are set and `numpy.ndarray`. The ``mask`` is
        padded with ones, assuming that the new elements should be masked
        because they contain no relevant data::

            >>> nddo.mask
            array([[ True,  True,  True,  True,  True],
                   [ True, False, False, False,  True],
                   [ True, False, False, False,  True],
                   [ True, False, False, False,  True],
                   [ True,  True,  True,  True,  True]], dtype=bool)

        If you specifiy a tuple as offset it is applied as padding to all
        dimensions, in this case: pad with one element in front and 2 at the
        end::

            >>> nddo = ndd.offset((1, 2))
            >>> nddo
            NDData([[0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0, 0],
                    [0, 1, 1, 1, 0, 0],
                    [0, 1, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0]])

        or explicitly giving it for all dimensions::

            >>> nddo = ndd.offset(((1, 2), (2, 1)))
            >>> nddo
            NDData([[0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0]])
        """
        # Abort slicing if the data is a single scalar.
        if self.data is None or self.data.shape == ():
            raise TypeError('scalars and None data cannot be offsetted.')

        pad_width_iterable = isinstance(pad_width, Iterable)

        # To avoid special casing the "pad_width" in the different offsetting
        # methods - could go crazy with wcs or if some attribute doesn't have
        # the same dimensions as the data. Better to catch that here before
        # doing the special cases there. See np.pad for a more extensive
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
        # Pad with zeros, let it fail if not possible. If the data is not
        # offsettable which attribute should?
        return pad(self.data, pad_width, mode='constant', constant_values=0)

    def _offset_uncertainty(self, pad_width):
        # Only pad if it's not None
        if self.uncertainty is None:
            return None
        # Try calling offset method. Uncertainty should implement it.
        try:
            return self.uncertainty.offset(pad_width)
        except (AttributeError, ValueError):
            pass
        # Returns the current value in case we couldn't pad it
        log.info("uncertainty cannot be offsetted.")
        return self.uncertainty

    def _offset_mask(self, pad_width):
        # Only pad if it's not None
        if self.mask is None:
            return None
        # Maybe it is a numpy.ndarray? Try to pad it.
        try:
            return pad(self.mask, pad_width, mode='constant',
                       constant_values=1)
        except ValueError:
            pass
        # Returns the current value in case we couldn't pad it
        log.info("mask cannot be offsetted.")
        return self.mask

    def _offset_wcs(self, pad_width):
        # Only pad if it's not None
        if self.wcs is None:
            return None
        # Check for a WCS instance, there we would need to alter the crpix
        # values.
        elif isinstance(self.wcs, WCS):
            wcs = self.wcs.deepcopy()
            # wcs naxis are reversed numpy axis, so we need to add the first
            # axis with the last "before"-padding
            wcs.wcs.crpix = [val + pad_width[-idx][0]
                             for idx, val in enumerate(wcs.wcs.crpix, 1)]
            return wcs
        # Maybe it was a numpy.ndarray? Try to pad it.
        else:
            try:
                return pad(self.wcs, pad_width, mode='constant',
                           constant_values=0)
            except ValueError:
                pass
        # Returns the current value in case we couldn't pad it
        log.info("wcs cannot be offsetted.")
        return self.wcs

    def _offset_flags(self, pad_width):
        # Only pad if it's not None
        if self.flags is None:
            return None
        # Maybe it is a numpy.ndarray? Try to pad it.
        try:
            return pad(self.flags, pad_width, mode='constant',
                       constant_values=0)
        except ValueError:
            pass
        # Returns the current value in case we couldn't pad it
        log.info("flags cannot be offsetted.")
        return self.flags
