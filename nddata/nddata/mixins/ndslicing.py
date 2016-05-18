# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from astropy import log


__all__ = ['NDSlicingMixin']


class NDSlicingMixin(object):
    """
    Mixin to provide slicing on objects using the `NDData`
    interface.

    The ``data``, ``mask``, ``uncertainty``, ``flags`` and ``wcs`` will be
    sliced, if set and sliceable. The ``unit`` and ``meta`` will be untouched.

    .. warning::
        The sliced return will be, if possible, a reference and not a copy.
    """
    def __getitem__(self, item):
        # Abort slicing if the data is a single scalar.
        if self.data.shape == ():
            raise TypeError('scalars cannot be sliced.')

        # Let the other methods handle slicing.
        kwargs = self._slice(item)
        return self.__class__(**kwargs)

    def _slice(self, item):
        """
        Collects the sliced attributes and passes them back as `dict`.

        It passes uncertainty, mask and wcs to their appropriate ``_slice_*``
        method, while ``meta`` and ``unit`` are simply taken from the original.
        The data is assumed to be sliceable and is sliced directly.

        When possible the return should *not* be a copy of the data but a
        reference.

        Parameters
        ----------
        item : slice
            The slice passed to ``__getitem__``.

        Returns
        -------
        dict :
            Containing all the attributes after slicing - ready to
            use them to create ``self.__class__.__init__(**kwargs)`` in
            ``__getitem__``.
        """
        kwargs = {}
        kwargs['data'] = self.data[item]
        # Try to slice some attributes
        kwargs['uncertainty'] = self._slice_uncertainty(item)
        kwargs['mask'] = self._slice_mask(item)
        kwargs['wcs'] = self._slice_wcs(item)
        kwargs['flags'] = self._slice_flags(item)
        # Attributes which are copied and not intended to be sliced
        kwargs['unit'] = self.unit
        kwargs['meta'] = self.meta
        return kwargs

    def _slice_uncertainty(self, item):
        if self.uncertainty is None:
            return None
        try:
            return self.uncertainty[item]
        except TypeError:
            # Catching TypeError in case the object has no __getitem__ method.
            # But let IndexError raise.
            log.info("uncertainty cannot be sliced.")
        return self.uncertainty

    def _slice_mask(self, item):
        if self.mask is None:
            return None
        try:
            return self.mask[item]
        except TypeError:
            log.info("mask cannot be sliced.")
        return self.mask

    def _slice_wcs(self, item):
        if self.wcs is None:
            return None
        try:
            return self.wcs[item]
        except TypeError:
            log.info("wcs cannot be sliced.")
        return self.wcs

    def _slice_flags(self, item):
        if self.flags is None:
            return None
        try:
            return self.flags[item]
        except TypeError:
            log.info("flags cannot be sliced.")
        return self.flags
