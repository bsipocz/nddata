# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from ..nduncertainty_var import VarianceUncertainty
from ...utils.numpyutils import expand_multi_dims


__all__ = ['NDReduceMixin']


class NDReduceMixin(object):
    """Mixin to provide methods for `~nddata.nddata.NDDataBase` which are \
            applied along one dimension (axis) of the data.

    These methods take the ``mask`` and ``uncertainty`` besides the ``data``
    into account.
    """

    def _reduce_get_mask(self):
        """Mostly for subclasses that don't use numpy bool masks as "mask".

        This function should return ``None`` or a `numpy.ndarray` of boolean
        type that can be used for boolean indexing. This function takes no
        arguments but can use every attribute of the instance it wants.

        See also
        --------
        NDStatsMixin._stats_get_mask
        NDClippingMixin._clipping_get_mask

        Notes
        -----
        See NDClippingMixin._clipping_get_mask for more context why this
        method works like this. It is essential for efficiency to create a
        boolean array here in case no mask is set.
        """
        if isinstance(self.mask, np.ndarray) and self.mask.dtype == bool:
            return self.mask
        return np.zeros(self.data.shape, dtype=bool)

    def reduce_average(self, axis=0, weights=None):

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
        axis = int(abs(axis))

        # Get the data and the mask from the instance attributes
        data = self.data
        mask = self._reduce_get_mask()

        # Setup the masked array based on the data and mask saved in the
        # instance.
        marr = np.ma.array(data, mask=mask, copy=False)

        # Abort the call in case the array is 1D, for 1D statistics see the
        # NDStatsMixin.
        if marr.ndim < 2:
            raise ValueError('reduce functions need the data to have more '
                             'than one dimension.')

        # Calculate the reduced data with np.average. The weights will be
        # checked in here and an appropriate exception is raised if the shape
        # does not match.
        red_data = np.ma.average(marr, axis=axis, weights=weights)

        # There is no builtin ufunc to calculate the weighted standard
        # deviation so we need to do use the average again. This will
        # calculate the variance of the average, but we have a
        # VarianceUncertainty and the user can convert it later if he wants
        # standard deviations.
        # To calculate the difference we need to expand the reduced dimension
        # of the reduced data again otherwise broadcasting could fail.
        diff = marr - np.expand_dims(red_data, axis=axis)
        red_uncertainty = np.ma.average(diff**2, axis=axis, weights=weights)

        # To get the variance of the mean we need to divide this reduced
        # variance by the number of valid values. This requires that we
        # manually change the weights now so they broadcast correctly against
        # the mask of the data. This is only required if the weights don't have
        # the same shape as the data
        if weights.shape != marr.shape:
            # Add empty dimensions to the weights that are not along the axis
            # along which the average was applied.
            weights = expand_multi_dims(weights, axis=axis, ndims=marr.ndim)

        # Determine the number of valid values we have in the end for each
        # resulting value. This can be calculated by multiplying the number of
        # valid pixel (inverse mask "~mask") with the weights. And then summing
        # along the axis we have reduced the data
        n_values = (~marr.mask * weights).sum(axis=axis)

        # So we don't end up with division by 0 problems set the values where
        # we have no valid value to 1. Since the average of the uncertainty
        # contains zeros where no valid element was present the corrected
        # variance will be calculated there as 0/1 = 0 which is exactly what
        # we would expect.
        no_valid_value = (n_values == 0)
        # TODO: Make sure np.ma.average really returns 0 when no valid element
        # was found along that axis!
        n_values[no_valid_value] = 1

        # To get the variance of the mean we divide by the number of valid
        # elements.
        red_uncertainty = VarianceUncertainty(red_uncertainty / n_values)
        # TODO: In theory it could be that we need some bias (dof) correction
        # here. So either allow a ddof parameter here or clearly state that
        # this isn't done here!

        # TODO: The number of valid elements would make a good flag array
        # maybe include it?

        # The "red_data" is a masked array so the resulting class should
        # split data and mask by itself.
        return self.__class__(red_data, uncertainty=red_uncertainty)
