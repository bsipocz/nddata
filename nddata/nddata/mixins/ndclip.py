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
        None and will overwrite the current mask with the resulting mask.
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
                           copy=False)

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

        # This approach uses advanced indexing like it is used if you index
        # with the result of a np.where() call. We create mgrids for each axis
        # except the one we compute along and insert the computed indices at
        # the appropriate place when we look for the highest/lowest elements.

        # Create a grid to index the elements. The indexes for the highest
        # value will be inserted later so we need only setup the other axis.
        # It is crucial that we special case 1 and 2 dimensions here because
        # the iteration would otherwise go over the grid and not over the axis.
        if marr.ndim == 1:
            # Empty indexes, because we only need to index along the axis
            # we compute the positions of the maxima.
            idx = []
        elif marr.ndim == 2:
            # This case is special because we create a 1D grid and the
            # iteration I've used for 3d+ would iterate over the 1D grid
            # and yield a wrong result. So we just create a list containing
            # the one grid and insert the grid of the axis that WAS NOT
            # specified.
            idx = [np.mgrid[slice(marr.shape[0 if axis == 1 else 1])]]
        else:
            # Create a list of the grids over all axis except the one that was
            # specified.
            idx = [i for i in np.mgrid[[slice(marr.shape[ax])
                                        for ax in range(marr.ndim)
                                        if ax != axis]]]

        # Start finding and clipping the lowest values
        for i in range(nlow):
            # Finding the coordinates with np.ma.argmin along the axis
            minCoord = np.ma.argmin(marr, axis=axis)
            # Insert these indexes at the "axis"-position of the index grids.
            idx.insert(axis, minCoord)
            # Set all these elements to masked (True)
            marr.mask[idx] = True
            # And remove the positions of the minima again. This is much faster
            # than creating a new list of mgrids each iteration.
            del idx[axis]

        # Same for the highest values for each pixel
        for i in range(nhigh):
            maxCoord = np.ma.argmax(marr, axis=axis)
            idx.insert(axis, maxCoord)
            marr.mask[idx] = True
            del idx[axis]

        # We just replace the mask. This could be problematic in case someone
        # wanted the mask to be something else but that shouldn't be the most
        # common case.
        self.mask = marr.mask
        return None  # explicitly return None, could also be omitted.

    def _clip_extrema_old(self, nlow=0, nhigh=0, axis=None):
        """This is the original version of the "clip_extrema" method. In case
        some weird constellation shows that this is faster than the new method
        I'll leave it in here for now.

        TODO: Benchmark and then remove this or make it public again.
        """
        nlow = int(abs(nlow))
        nhigh = int(abs(nhigh))
        if axis is not None:
            axis = int(abs(axis))

        if nlow == 0 and nhigh == 0:
            return None

        marr = np.ma.array(self.data, mask=self._clipping_get_mask(),
                           copy=False, hard_mask=False)

        if axis is None:
            for i in range(nlow):
                idx = np.ma.argmin(marr)
                marr.mask.ravel()[idx] = True
            for i in range(nhigh):
                idx = np.ma.argmax(marr)
                marr.mask.ravel()[idx] = True

            self.mask = marr.mask
            return None
        if axis >= marr.ndim:
            raise IndexError('cannot clip along non-existent axis.')

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

        self.mask = marr.mask

        return None

    def _clipping_get_mask(self):
        """Mostly for subclasses that don't use numpy bool masks as "mask".

        This function should return ``None`` or a `numpy.ndarray` of boolean
        type that can be used for boolean indexing. This function takes no
        arguments but can use every attribute of the instance it wants.

        See also
        --------
        NDStatsMixin._stats_get_mask

        Notes
        -----
        It is very important that you do not return "None" here because
        np.ma.array(self.data, mask=None) is MUCH MUCH MUCH more slower than
        np.ma.array(self.data, mask=np.zeros(self.data.shape, dtype=bool)).
        I measured 800 ms for a (100, 100, 100) array with None vs. 450 us with
        np.zeros. This is more than a factor of 1000!
        See also:
        http://stackoverflow.com/questions/37468069/why-is-np-ma-array-so-slow-with-mask-none-or-mask-0
        """
        if isinstance(self.mask, np.ndarray) and self.mask.dtype == bool:
            return self.mask
        return np.zeros(self.data.shape, dtype=bool)
        # numpy 1.11 also special cases False and True but not before, so this
        # function is awfully slow then.
        # return False
