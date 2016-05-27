# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

__all__ = ['NDClippingMixin']


class NDClippingMixin(object):
    """Mixin class to add clipping methods to `nddata.nddata.NDDataBase`.

    .. warning::
        These methods do require the mask to be a boolean `numpy.ndarray` or
        None and will overwrite the current mask with the resulting mask.
    """

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

    def clip_extrema(self, nlow=0, nhigh=0, axis=None):
        """Clip the lowest and/or highest values along an axis.

        .. note::
            It is possible to call this method without providing parameters but
            then this method doesn't do anything.

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

        Raises
        ------
        IndexError
            In case the ``axis`` given doesn't exist in the data.

        Notes
        -----
        The input parameters are cast to the restrictions, so invalid inputs
        may not trigger an Exception **but** may yield unexpected results. For
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
            array([ True,  True, False, False, False, False, False,  True], \
dtype=bool)

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

        As all of the other clipping methods this will take the original mask
        into account::

            >>> data = np.array([0, 1, 2, 3])
            >>> mask = data > 1
            >>> ndd = NDData(data, mask=mask)
            >>> ndd.clip_extrema(nhigh=1)
            >>> ndd.mask
            array([False,  True,  True,  True], dtype=bool)

        Here the values 3 and 2 are already masked so the highest remaining
        element was 1.
        """

        # nlow and nhigh are the number of lowest or highest points to be
        # masked. We can only mask whole values and we cannot handle negative
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

        # Same for the highest values
        for i in range(nhigh):
            maxCoord = np.ma.argmax(marr, axis=axis)
            idx.insert(axis, maxCoord)
            marr.mask[idx] = True
            del idx[axis]

        # Overwrite the saved mask and return None
        self.mask = marr.mask
        return None  # explicitly return None, could also be omitted.

    def _clip_extrema_old(self, nlow=0, nhigh=0,
                          axis=None):  # pragma: no cover
        """This is the original version of the "clip_extrema" method. In case
        some weird constellation shows that this is faster than the new method
        I'll leave it in here for now.

        TODO: Benchmark and then either remove this or make it public again.
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

        # Same for the highest values
        for i in range(nhigh):
            maxCoord = np.expand_dims(np.ma.argmax(marr, axis=axis), axis=axis)
            newmask = cmp == maxCoord
            marr.mask |= newmask

        self.mask = marr.mask

        return None

    def clip_invalid(self):
        """Clip elements that are ``NaN`` or ``Inf``.

        See also
        --------
        numpy.isfinite

        Examples
        --------
        Consider you have an array where invalid values like ``NaN`` or ``Inf``
        are present. These can be masked by using clip_invalid::

            >>> from nddata.nddata import NDData
            >>> import numpy as np

            >>> ndd = NDData([1, np.nan, np.inf, -np.inf, 10])
            >>> ndd.clip_invalid()
            >>> ndd.mask
            array([False,  True,  True,  True, False], dtype=bool)
        """
        # Get the current mask.
        mask = self._clipping_get_mask()
        # Combine the mask by using a logical_or. The most appropriate way
        # of finding invalid values is by using the inverse of np.isfinite.
        # np.isfinite return True for each value that is not NaN or not Inf so
        # we need to invert it.
        mask |= ~np.isfinite(self.data)

        # Overwrite the saved mask and return None
        self.mask = mask
        return None

    def clip_sigma(self, sigma=3, sigma_lower=None, sigma_upper=None,
                   iters=None, cenfunc=np.ma.median, stdfunc=np.std,
                   axis=None):
        """Perform sigma-clipping on the provided data.

        The data will be iterated over, each time rejecting points that are
        discrepant by more than a specified number of standard deviations from
        a center value. Invalid values (NaNs or Infs) are automatically masked
        before performing the sigma clipping.

        See also
        --------
        astropy.stats.sigma_clip
        scipy.stats.sigmaclip

        Parameters
        ----------
        sigma : number, optional
            The number of standard deviations to use for both the lower and
            upper clipping limit. These limits are overridden by
            ``sigma_lower`` and ``sigma_upper``, if given.
            Default is ``3``.

        sigma_lower : number or None, optional
            The number of standard deviations to use as the lower bound for
            the clipping limit. If ``None`` then the value of ``sigma`` is
            used.
            Default is ``None``.

        sigma_upper : number or None, optional
            The number of standard deviations to use as the upper bound for
            the clipping limit. If ``None`` then the value of ``sigma`` is
            used.
            Default is ``None``.

        iters : positive `int` or None, optional
            The number of iterations to perform sigma clipping, or ``None`` to
            clip until convergence is achieved (i.e., continue until the
            iteration masks no further values).
            Default is ``None``.

        cenfunc : callable, optional
            The function used to compute the center for the clipping. Must
            be a callable that can handle `numpy.ma.MaskedArray` and outputs
            the central value.

            Recommended functions are:

            - :func:`numpy.ma.median` : median
            - :func:`numpy.mean` : mean (can handle masked arrays)

            Default is :func:`numpy.ma.median`.

        stdfunc : callable, optional
            The function used to compute the standard deviation about the
            center. Must be a callable that can handle `numpy.ma.MaskedArray`
            and outputs a width estimator. Values are rejected based on::

                 deviation < (-sigma_lower * stdfunc(deviation))
                 deviation > (sigma_upper * stdfunc(deviation))

            and::

                deviation = data - cenfunc(data [,axis=int])

            Recommended functions are:

            - :func:`numpy.std` : standard deviation (can handle masked arrays)
            - :func:`~astropy.stats.mad_std` : median standard deviation (can \
                handle masked arrays since astropy 1.0.9)

            Default is :func:`numpy.std`.

        axis : `int` or None, optional
            If not ``None``, clip along the given axis. For this case,
            ``axis`` will be passed on to ``cenfunc`` and ``stdfunc``, which
            are expected to return an array with the axis dimension removed
            (like the numpy functions). If ``None``, clip over all axes.
            Default is ``None``.

        Examples
        --------
        To clip values that deviate by three sigma (in terms of standard
        deviation) from the median of the data::

            >>> import numpy as np
            >>> from nddata.nddata import NDData

            >>> ndd = NDData([0, 10, 9, 10, 11, 9, 10, 10, 11])
            >>> ndd.clip_sigma(cenfunc=np.ma.median, stdfunc=np.ma.std,
            ...                sigma=3)
            >>> ndd.mask
            array([ True, False, False, False, False, False, False, False, \
False], dtype=bool)

        .. note::
            This method is inspired by ``astropy.stats.sigma_clip()`` but
            modified in several ways.
        """
        # TODO: As soon as older astropy versions aren't supported anymore
        # one could just wrap sigma_clip. Except for the iteration end
        # condition, the negative axis and the slightly different build the
        # functionality is identical. It's just that it underwent major changes
        # between astropy 1.0, 1.1 and 1.2 and not every version of astropy is
        # built against older numpy versions and for now I feel it's better to
        # support numpy 1.7 and 1.8 than supporting older astropy versions.

        # The upper and lower sigma are the parameters that are used internally
        # so check if they are explicitly provided or use the general sigma
        # value. Because we work with positive defined sigma values later we
        # take the precaution to use the absolute in case someone entered for
        # example a negative value for the lower sigma.
        sigma_lower = abs(sigma) if sigma_lower is None else abs(sigma_lower)
        sigma_upper = abs(sigma) if sigma_upper is None else abs(sigma_upper)

        # We allow the iterations to be None. To avoid duplication of the logic
        # during the iterations of clipping we just set it to np.inf. This
        # will ensure that the iterations run as long as the break condition
        # is not met. DO NOT FORGET THE BREAK CONDITION!
        # If the iters value is not None we expect it to be a positive integer
        # so it's cast to one in that case.
        iters = np.inf if iters is None else int(abs(iters))

        # we expect only a single given axis or None as axis and because the
        # expand-dims later wouldn't work with multiple axis we convert it to
        # an integer here. But we allow the choice of it being negative, maybe
        # it works.
        if axis is not None:
            axis = int(axis)

        # Clip invalid points so that the function calculating the center and
        # deviation do not need to account for NaNs and the mask. This is done
        # before the temporary masked array is created.
        self.clip_invalid()

        # Create a masked array on which to perform the centering and deviation
        # determination. Based on the current data and mask.
        marr = np.ma.array(self.data, mask=self._clipping_get_mask(),
                           copy=False)

        # The initial count of unmasked elements. This is used as break
        # condition for the iterations. But we calculate only the counts of
        # unmasked elements AFTER the clipping during the iterations so we need
        # to calculate the initial count before we start iterating.
        n_nomask_before = marr.count()

        # Clip as long as the number of masked values change AND the number of
        # specified iterations isn't exhausted
        while iters > 0:
            # decrement the iterations counter. This has no effect on np.inf so
            # the iters=None case will iterate forever.
            iters -= 1

            # Calculate the center using the center-func. This variable will
            # become the upper threshold later to avoid creating an additional
            # intermediate array. Thus we name it u_threshold.
            u_threshold = cenfunc(marr, axis=axis)
            # Calculate the value for standard deviation using the stdfunc.
            dev = stdfunc(marr, axis=axis)
            # Create a new array for the lower threshold
            l_threshold = u_threshold - sigma_lower * dev
            # For the upper threshold we use the array where we stored the
            # center and add the deviation criterion INPLACE.
            u_threshold += sigma_upper * dev

            # If we specified an axis we need to add the original axis back
            # to ensure correct broadcasting later. We actuall do not need to
            # do that for axis=0 but expand_dims is rather cheap especially if
            # it is done in place. So apart from memory reasons there is no
            # need to special case ONE possible axis value. But only expand the
            # dimensions IF the original array has more than one dimension.
            if axis is not None and marr.ndim > 1:
                l_threshold = np.expand_dims(l_threshold, axis=axis)
                u_threshold = np.expand_dims(u_threshold, axis=axis)

            # Mask the values below and above the thresholds inplace.
            marr.mask |= marr.data < l_threshold
            marr.mask |= marr.data > u_threshold

            # Calculate the number of unmasked elements in the array and
            # compare this value to the last count. If the value has not
            # changed further iterations do not make sense so we break the
            # loop. But if they differ use the new count as the new counts
            # before value so we don't need to calculate the counts twice in
            # each iteration.
            n_nomask_after = marr.count()
            if n_nomask_after == n_nomask_before:
                break
            n_nomask_before = n_nomask_after

        # After the clipping iterations just take the mask of the intermediate
        # masked array and set it as new mask attribute.
        self.mask = marr.mask
        return None

    def clip_range(self, low=None, high=None):
        """Clip elements that are outside a given range.

        .. note::
            It is possible to call this method without providing parameters but
            then this method doesn't do anything.

        Parameters
        ----------
        low : number or None, optional
            The minimal valid value. Values which are smaller than this
            parameter will be masked. ``None`` disables this lower bound
            criterion and no value will be masked because of it.
            Default is ``None``.

        high : number or None, optional
            The maximal valid value. Values which are greater than this
            parameter will be masked. ``None`` disables this lower bound
            criterion and no value will be masked because of it.
            Default is ``None``.

        Examples
        --------
        One can give the lower and higher threshold like this::

            >>> import numpy as np
            >>> from nddata.nddata import NDData

            >>> data = np.arange(10)
            >>> ndd = NDData(data)
            >>> ndd.clip_range(low=2.1, high=4)
            >>> ndd.mask
            array([ True,  True,  True, False, False,  True,  True,  True,  \
True,  True], dtype=bool)

        .. note::
            This method is inspired by ``ccdproc.Combiner.minmax_clipping()``
            and only minimally modified.
        """
        # If neither parameter is set exit this method immediatly, nothing to
        # be done in here.
        if low is None and high is None:
            return None

        # In any case we need the current mask.
        mask = self._clipping_get_mask()

        # Mask values below or above the threshold if any of these parameters
        # are set. Then make an in-place logical_or to propagate these values.
        if low is not None:
            mask |= self.data < low
        if high is not None:
            mask |= self.data > high

        # Overwrite the saved mask and return None
        self.mask = mask
        return None
