# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from collections import OrderedDict

import numpy as np

from astropy import log
from astropy.table import Table
from astropy.stats import mad_std, biweight_location, biweight_midvariance

from ...utils.stats import mode

from ... import OPT_DEPS
if OPT_DEPS['SCIPY']:  # pragma: no cover
    from scipy.stats import skew, kurtosis

__all__ = ['NDClippingMixin']


class NDClippingMixin(object):
    """Mixin class to add clipping methods to `nddata.nddata.NDDataBase`.

    .. warning::
        These methods do require the mask to be a boolean `numpy.ndarray` or
        None and will produce such masks.
    """

    def clip_extrema(self, nlow=0, nhigh=0, axis=None):
        """Clip the lowest and/or highest values along an axis.

        Parameters
        ----------
        nlow, nhigh : positive `int`, optional
            The number of lowest (``nlow``) or highest (``nhigh``) values to
            mask along each axis.
            Default is ``0``.

        axis : positive `int` or None, optional
            The axis along which to perform the clipping. This follows the
            ``NumPy`` standards of 0-based indexing, meaning that ``axis=0``
            is the first axis. Setting it to ``None`` will perform the clipping
            over the whole data, not along any axis.

            Default is ``None``.

        Returns
        -------
        nothing : `None`
            This method will update the ``mask`` attribute but returns `None`.

        Raises
        ------
        IndexError
            In case the ``axis`` given doesn't exist in the data.

        Notes
        -----
        The input parameters are cast to the restrictions, so invalid inputs
        may not trigger an Exception **but** yield unexpected results. For
        example the ``nlow`` parameter is cast to a positive integer by:
        ``nlow = int(abs(nlow))`` and similar for ``nhigh`` and ``axis``.

        The first occurence of the minimum an maximum is masked in case two
        values are identical.

        Examples
        --------

        To clip the lowest and the two highest values from a one-dimensional
        array::

            >>> from nddata.nddata import NDData
            >>> ndd = NDData([7,1,3,2,5,1,5,7])
            >>> ndd.clip_extrema(nlow=1, nhigh=2)
            >>> ndd.mask
            array([ True,  True, False, False, False, False, False,  True], dtype=bool)

        Only the first occurence of the lowest value (1) was masked, because
        the ``nlow`` parameter only masks one value even if it is contained
        multiple times.

        The ``axis`` parameter also allows to clip along arbitary axis in a
        multidimensional array instead of over the whole array::

            >>> ndd = NDData([[7,1,3],[2,5,1],[5,7,3]])
            >>> ndd.clip_extrema(nhigh=1, axis=1)
            >>> ndd.mask
            array([[ True, False, False],
                   [False,  True, False],
                   [False,  True, False]], dtype=bool)
        """

        # nlow and nhigh are the number of lowest or highest points to be
        # masked. We can only mask whole pixel and we cannot handle negative
        # values so take the absolute and cast to integer.
        nlow = int(abs(nlow))
        nhigh = int(abs(nhigh))

        # In case nlow and nhigh are zero we have nothing to do.
        if nlow == 0 and nhigh == 0:
            return None

        # If an axis is given make sure it's an integer as well, we cannot
        # operate on multiple axis or fractional axis. Negative axis don't work
        # yet because of the following expand_dims during clipping.
        if axis is not None:
            axis = int(abs(axis))

        # We will work with masked arrays and their methods here so we have
        # to create one. Get the mask from a customizable function so
        # subclasses can override it.
        marr = np.ma.array(self.data, mask=self._clipping_get_mask(),
                           copy=False, hard_mask=False)

        # If the axis is None the procedure differs from integer axis cases
        # because we don't need the expand dims and comparison array, so handle
        # this case seperate
        if axis is None:
            for i in range(nlow):
                # Get the index where the global minimum of the masked array is
                # but beware that this is the index of the ravelled array, so
                # we need to set the appropriate ravelled mask attribut to True
                idx = np.ma.argmin(marr)
                marr.mask.ravel()[idx] = True
            for i in range(nhigh):
                idx = np.ma.argmax(marr)
                marr.mask.ravel()[idx] = True

            # Replace the mask of the NDDataBase instance with the new mask,
            # since we included the original mask (and if the user doesn't use
            # threads for multiprocessing) this mask represents the new mask.
            self.mask = marr.mask
            # explicitly return None, could be omitted but then the other case
            # needs to be in an else clause.
            return None

        # The following only applies to an axis that is integer and not None!

        # In case the axis is bigger than the number of dimensions we cannot
        # proceed so give them an IndexError.
        if axis >= marr.ndim:
            raise IndexError('cannot clip along non-existent axis.')

        # TODO: Maybe make axis=None work as well, but keep it like is for now.

        # This comparison array is just an array containing the indices along
        # the axis we want to compare. In case the axis is not the last
        # dimension of the array we need to expand it's dimensions so it
        # matches the array. Last axis works because numpy broadcasting makes
        # it work quite easily.
        cmp = np.arange(marr.shape[axis])
        if axis != marr.ndim - 1:  # last axis doesn't need this
            for i in (i for i in range(marr.ndim) if i != axis):
                cmp = np.expand_dims(cmp, axis=i)

        # Start finding and clipping the lowest values
        for i in range(nlow):
            # Find the coordinate in axis direction where the minimum value is
            # argmin will return an array with one dimension less than the
            # original array and therefore broadcasting would only work if we
            # were along the last axis. Adding to the problems numpy.ma ufuncs
            # do not have a keepdims parameter but we can emulate it by
            # expanding the dimensions afterwards again.
            minCoord = np.expand_dims(np.ma.argmin(marr, axis=axis), axis=axis)
            # To get a valid mask we need to get this min-coordinates to the
            # same shape as the original mask. I'll do that by using numpy
            # broadcasting. This will create a mask where everything is False
            # except where the mincoordinates match the cmp-array. This will
            # produce the required result. But also create another mask with
            # the same shape as the original masks. Bools are quite cheap and
            # the data will probably be the one eating the memory but in case
            # we are close to the memory limit already this could push it into
            # swap memory.
            newmask = cmp == minCoord
            # Now just update the mask of the masked array.
            marr.mask |= newmask

        # Same for the highest values for each pixel
        for i in range(nhigh):
            maxCoord = np.expand_dims(np.ma.argmax(marr, axis=axis), axis=axis)
            newmask = cmp == maxCoord
            marr.mask |= newmask

        # We just replace the mask. This could be problematic in case someone
        # wanted the mask to be something else but that shouldn't be the most
        # common case.
        self.mask = marr.mask

        return None  # explicitly return None, could also be omitted.

    def _clipping_get_mask(self):
        """Mostly for subclasses that don't use numpy bool masks as "mask".

        This function should return ``None`` or a `numpy.ndarray` of boolean
        type that can be used for boolean indexing. This function takes no
        arguments but can use every attribute of the instance it wants.

        See also
        --------
        NDStatsMixin._stats_get_mask
        """
        if isinstance(self.mask, np.ndarray) and self.mask.dtype == bool:
            return self.mask
        return None
