# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from ...utils.copyutils import do_copy

import numpy as np

from ..nduncertainty_stddev import StdDevUncertainty
from ..nduncertainty_var import VarianceUncertainty
from ...utils.inputvalidation import as_unsigned_integer


__all__ = ['NDReduceMixin']


class NDReduceMixin(object):
    """Mixin to provide methods for `~nddata.nddata.NDDataBase` which are \
            applied along one dimension (axis) of the data.

    These methods take the ``mask`` besides the ``data`` into account and
    calculate based on the error of the result.

    .. note::
        The ``unit`` and ``meta`` of the result will be a copy of the original
        `~nddata.nddata.NDDataBase` instance. ``wcs`` and ``flags`` as well but
        this might change because they **should** be subject to a reduction
        themselves- depending on the type of attribute.
    """

    def _reduce_get_others(self):
        # Meta and unit should stay the same for the reduce functions.
        kwargs = {'meta': do_copy(self.meta),
                  'unit': self.unit,
                  'wcs': do_copy(self.wcs),
                  'flags': do_copy(self.flags)}
        # TODO: WCS and Flags may also be subject to changes because of the
        # reduction, but currently just copy them.
        return kwargs

    def reduce_average(self, axis=0, weights=None):
        """Compute the average along an axis with specified weights.

        Parameters
        ----------
        axis: positive `int`, optional
            The axis (dimension) along which to compute the average. Must not
            be ``None``. If you are loooking for overall statistics use:
            :meth:`~nddata.nddata.mixins.NDStatsMixin.stats`.
            Default is ``0``.

        weights : `numpy.ndarray`-like or None, optional
            The weights for averaging. Must be scalar or have the same shape as
            the ``data`` or the same length as ``data.shape[axis]``. If the
            weights are ``None`` it will call :meth:`reduce_mean`.
            Default is ``None``.

        Returns
        -------
        ndd : `~nddata.nddata.NDDataBase`-like
            The result will have the same class as the instance this method was
            called on. The results ``data`` contains the average of the
            calculation while the ``mask`` is set in case any element had no
            values to average and the ``uncertainty`` will be the variance of
            the average (already corrected by the number of valid elements).

        Examples
        --------

        Calculate the weighted mean of a 2 x 5 array along the first axis::

            >>> import numpy as np
            >>> from nddata.nddata import NDData

            >>> ndd = NDData([[3, 2, 1, 1, 4], [2, 2, 2, 2, 2]],
            ...              mask=np.array([[0, 1, 0, 1, 0], [0, 1, 0, 0, 0]],
            ...                            dtype=bool))
            >>> avg = ndd.reduce_average(axis=0, weights=[1, 1.5])
            >>> avg
            NDData([ 2.4,  0. ,  1.6,  2. ,  2.8])
            >>> avg.mask
            array([False,  True, False, False, False], dtype=bool)
            >>> avg.uncertainty
            VarianceUncertainty([ 0.096,  0.   ,  0.096,  0.   ,  0.384])

        .. note::
            The correction for the resulting uncertainty is the total number of
            valid values **without** taking any degrees of freedom into
            account.
        """
        # If no weights are given this is essentially a mean reduce. So return
        # the mean reduction result.
        if weights is None:
            return self.reduce_mean(axis=axis)

        # To allow also list-like weights convert them to a numpy array here.
        # Since this doesn't copy existing np.arrays this is relativly cheap if
        # it's already an array.
        weights = np.asarray(weights)

        # The axis must be integer and because of later restrictions it also
        # needs to be positive.
        axis = as_unsigned_integer(axis)

        # Get the data and the mask from the instance attributes
        data = self.data
        mask = self._get_mask_numpylike()

        # Setup the masked array based on the data and mask saved in the
        # instance. Important profiling information about this np.any is
        # described in reduce_mean. This should stay the way it is.
        if np.any(mask):
            marr = np.ma.array(data, mask=mask, copy=False)
            avg_func = np.ma.average
        else:
            marr = np.array(data, copy=False)
            avg_func = np.average

        # Abort the call in case the array is 1D, for 1D statistics see the
        # NDStatsMixin.
        if marr.ndim < 2:
            raise ValueError('reduce functions need the data to have more '
                             'than one dimension.')

        # Calculate the reduced data with np.average. The weights will be
        # checked in here and an appropriate exception is raised if the shape
        # does not match.
        red_data = avg_func(marr, axis=axis, weights=weights)

        # There is no builtin ufunc to calculate the weighted standard
        # deviation so we need to do use the average again. This will
        # calculate the variance of the average, but we have a
        # VarianceUncertainty and the user can convert it later if he wants
        # standard deviations.
        # To calculate the difference we need to expand the reduced dimension
        # of the reduced data again otherwise broadcasting could fail.
        diff = (marr - np.expand_dims(red_data, axis=axis)) ** 2
        red_uncert, eff_weights = avg_func(diff, axis=axis, weights=weights,
                                           returned=True)

        # To get the variance of the mean we need to divide this reduced
        # variance by the number of valid values. This number of valid values
        # are contained in the "eff_weights".

        # So we don't end up with division by 0 problems set the values where
        # we have no valid value to 1. Since the average of the uncertainty
        # contains zeros where no valid element was present - the corrected
        # variance will be calculated there as 0/1 = 0 which is exactly what
        # we would expect. And not the 0/0 = nan we would otherwise have.
        no_valid_value = (eff_weights == 0)
        eff_weights[no_valid_value] = 1

        # To get the variance of the mean we divide by the number of valid
        # elements.
        red_uncert = VarianceUncertainty(red_uncert / eff_weights)
        # TODO: In theory it could be that we need some bias (dof) correction
        # here. So either allow a ddof parameter here or clearly state that
        # this isn't done here!

        # TODO: The number of valid elements would make a good flag array
        # maybe include it?

        # The "red_data" is a masked array so the resulting class should
        # split data and mask by itself.
        return self.__class__(red_data, uncertainty=red_uncert,
                              **self._reduce_get_others())

    def reduce_mean(self, axis=0):
        """Compute the mean along an axis.

        Parameters
        ----------
        axis: positive `int`, optional
            The axis (dimension) along which to compute the mean. Must not
            be ``None``. If you are loooking for overall statistics use:
            :meth:`~nddata.nddata.mixins.NDStatsMixin.stats`.
            Default is ``0``..

        Returns
        -------
        ndd : `~nddata.nddata.NDDataBase`-like
            The result will have the same class as the instance this method was
            called on. The results ``data`` contains the mean of the
            calculation while the ``mask`` is set in case any element had no
            values to avergae and the ``uncertainty`` will be the variance of
            the mean (already corrected by the number of valid elements).

        Examples
        --------

        Calculate the mean of a 2 x 5 array along the first axis::

            >>> import numpy as np
            >>> from nddata.nddata import NDData

            >>> ndd = NDData([[3, 2, 1, 1, 4], [2, 2, 2, 2, 2]],
            ...              mask=np.array([[0, 1, 0, 1, 0], [0, 1, 0, 0, 0]],
            ...                            dtype=bool))
            >>> avg = ndd.reduce_mean(axis=0)
            >>> avg
            NDData([ 2.5,  0. ,  1.5,  2. ,  3. ])
            >>> avg.mask
            array([False,  True, False, False, False], dtype=bool)
            >>> avg.uncertainty
            VarianceUncertainty([ 0.125,  0.   ,  0.125,  0.   ,  0.5  ])

        .. note::
            This method is identical to :meth:`reduce_average` with
            ``weights=None``.

        .. note::
            The correction for the resulting uncertainty is the total number of
            valid values **without** taking any degrees of freedom into
            account.
        """

        # Much the same as average but without weights and instead of average
        # with mean and std
        axis = as_unsigned_integer(axis)
        data = self.data
        mask = self._get_mask_numpylike()

        # np.mean and np.var work on masked arrays so can create a normal numpy
        # array if no value is masked. This will probably be a lot faster.

        # IMPORTANT: Line profiling shows that in case of big arrays the
        # _reduce_get_mask() function takes only 0.1% of the total run-time and
        # the np.any() 0-3% so this could make a difference if we special cased
        # the case when no mask is present but NOT much.
        # On the other hand the np.mean on a plain numpy array is approximatly
        # 6-10 times faster than on masked arrays so that actually makes a huge
        # difference. So even if we have a mask it could be wise to check if
        # there are any masked values at all.
        # Therefore: This should stay as is!
        if np.any(mask):
            marr = np.ma.array(data, mask=mask, copy=False)
            marr_is_masked = True
        else:
            marr = np.array(data, copy=False)
            marr_is_masked = False

        # Abort the call in case the array is 1D, for 1D statistics see the
        # NDStatsMixin.
        if marr.ndim < 2:
            raise ValueError('reduce functions need the data to have more '
                             'than one dimension.')

        red_data = np.mean(marr, axis=axis)

        # np.var and np.std have the same runtime but since we would need to
        # take the square root of the number of valid values calculating the
        # variance and then just dividing by the number of valid pixel is much
        # faster than calculating the std and then diving by the SQRT of the
        # number of valid pixel. In case someone wants the resulting
        # uncertainty in standard deviations he can cast it to one!
        red_uncertainty = np.var(marr, axis=axis)

        # We need to determine the number of valid pixel ourself, fortunatly
        # this number is just the sum of unmakes values along the specified
        # axis. With the correction for cases where no valid values is. This
        # correction is described in reduce_average.
        if marr_is_masked:
            n_values = (~marr.mask).sum(axis=axis)
            no_valid_value = (n_values == 0)
            n_values[no_valid_value] = 1
        else:
            # In case no values were masked the number of valid values is just
            # the length of the array along the given axis.
            n_values = marr.shape[axis]

        red_uncertainty = VarianceUncertainty(red_uncertainty / n_values)

        return self.__class__(red_data, uncertainty=red_uncertainty,
                              **self._reduce_get_others())

    def reduce_median(self, axis=0):
        """Compute the median along an axis.

        Parameters
        ----------
        axis: positive `int`, optional
            The axis (dimension) along which to compute the median. Must not
            be ``None``. If you are loooking for overall statistics use:
            :meth:`~nddata.nddata.mixins.NDStatsMixin.stats`.
            Default is ``0``..

        Returns
        -------
        ndd : `~nddata.nddata.NDDataBase`-like
            The result will have the same class as the instance this method was
            called on. The results ``data`` contains the median of the
            calculation while the ``mask`` is set in case any element had no
            values for the computation and the ``uncertainty`` will be the
            median absolute standard deviation of the median (already corrected
            by the number of valid elements).

        Examples
        --------
        Calculate the median of a 2 x 4 array along the first axis::

            >>> import numpy as np
            >>> from nddata.nddata import NDData

            >>> ndd = NDData([[3, 2, 1, 1], [2, 2, 2, 2]],
            ...              mask=np.array([[0, 1, 0, 1], [0, 1, 0, 0]],
            ...                            dtype=bool))
            >>> avg = ndd.reduce_median(axis=0)
            >>> avg
            NDData([ 2.5,  0. ,  1.5,  2. ])
            >>> avg.mask
            array([False,  True, False, False], dtype=bool)
            >>> avg.uncertainty
            StdDevUncertainty([ 0.52417904,  0.        ,  0.52417904,  0.     \
])

        .. note::
            The correction for the resulting uncertainty is the total number of
            valid values **without** taking any degrees of freedom into
            account.
        """

        # This method is some hybrid from average and mean reduce. Only the
        # real differences are commented upon. For further details on the
        # rationale see these other methods.
        axis = as_unsigned_integer(axis)
        data = self.data
        mask = self._get_mask_numpylike()

        if np.any(mask):
            marr = np.ma.array(data, mask=mask, copy=False)
            # np.median doesn't work on masked arrays so we need to use
            # np.ma.median here
            med_func = np.ma.median
            marr_is_masked = True
        else:
            marr = np.array(data, copy=False)
            med_func = np.median
            marr_is_masked = False

        if marr.ndim < 2:
            raise ValueError('reduce functions need the data to have more '
                             'than one dimension.')

        red_data = med_func(marr, axis=axis)

        # Constant is taken from astropy mad_std
        # IMPORTANT: Using the astropy.stats.mad_std would calculate the median
        # again, since we already have the median along the axis we can omit
        # this expensive recalculation - but then we cannot reuse mad_std. But
        # especially for large masked arrays the speed gain is huge.
        diff = np.abs(marr - np.expand_dims(red_data, axis=axis))
        red_uncertainty = 1.482602218505602 * med_func(diff, axis=axis)

        if marr_is_masked:
            n_values = (~marr.mask).sum(axis=axis)
            no_valid_value = (n_values == 0)
            n_values[no_valid_value] = 1
        else:
            n_values = marr.shape[axis]

        # This time we work with standard deviations because that's what
        # the median absolute deviation approximates so we need to take the
        # square root of the n_values correction factor
        n_values = np.sqrt(n_values)

        red_uncertainty = StdDevUncertainty(red_uncertainty / n_values)

        # FIXME: Strangly the result has an uncertainty different from 0 when
        # all values are masked here. This is not the case for average or mean
        # but it seems to be a problem with the median. I guess this is because
        # the np.expand_dims doesn't preserve the mask and something weird
        # happens so that the median of the "diff" doesn't realize it's all
        # masked and returns something. Maybe this could be a numpy Bug but for
        # now I just make it work by replacing them manually:
        if marr_is_masked:
            red_uncertainty.data[no_valid_value] = 0

        return self.__class__(red_data, uncertainty=red_uncertainty,
                              **self._reduce_get_others())
