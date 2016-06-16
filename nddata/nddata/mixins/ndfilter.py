# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from ..nduncertainty_var import VarianceUncertainty
from ..nduncertainty_stddev import StdDevUncertainty
from ...utils.numbautils import convolve, interpolate
from ...utils.numbautils import convolve_median, interpolate_median
from ...deps import OPT_DEPS

__all__ = ['NDFilterMixin']

if not OPT_DEPS['NUMBA']:  # pragma: no cover
    __doctest_skip__ = ['*']


class NDFilterMixin(object):
    """Mixin to allow some filtering (convolution, interpolation).
    """
    def interpolate_average(self, kernel):
        """Interpolate masked pixel in the instance by applying an average \
                filter.

        .. note::
            This function requires `Numba` and that the ``data`` has only 1, 2
            or 3 dimensions.

        Parameters
        ----------
        kernel : `numpy.ndarray`, `astropy.convolution.Kernel`
            The kernel (or footprint) for the interpolation. The sum of the
            ``kernel`` must not be 0 (or very close to it). Each axis of the
            kernel must be odd.

        Returns
        -------
        nothing : `None`
            The interpolation works in-place!

        Examples
        --------

        >>> from nddata.nddata import NDData
        >>> import numpy as np
        >>> ndd = NDData([1,100,1], mask=np.array([0,1,0], dtype=bool))
        >>> ndd.interpolate_average([1,1,1])
        >>> ndd
        NDData([ 1.,  1.,  1.])
        >>> ndd.mask
        array([False, False, False], dtype=bool)
        """
        self.data = interpolate(self.data, kernel, self._filter_get_mask())
        self.mask = np.isnan(self.data)

    def interpolate_median(self, kernel):
        """Interpolate masked pixel in the instance by applying an median \
                filter.

        .. note::
            This function requires `Numba` and that the ``data`` has only 1, 2
            or 3 dimensions.

        Parameters
        ----------
        kernel : `numpy.ndarray`, `astropy.convolution.Kernel`
            The kernel for the convolution. One difference from normal
            convolution is that the actual values of the kernel do not matter,
            except when it is zero then it won't use the element for the median
            computation. Each axis of the kernel must be odd.

        Returns
        -------
        nothing : `None`
            The interpolation works in-place!

        Examples
        --------

        >>> from nddata.nddata import NDData
        >>> import numpy as np
        >>> ndd = NDData([1,100,1], mask=np.array([0,1,0], dtype=bool))
        >>> ndd.interpolate_median([1,1,1])
        >>> ndd
        NDData([ 1.,  1.,  1.])
        >>> ndd.mask
        array([False, False, False], dtype=bool)
        """
        self.data = interpolate_median(self.data, kernel,
                                       self._filter_get_mask())
        self.mask = np.isnan(self.data)

    def filter_average(self, kernel, uncertainty=False):
        """Filter the instance data by applying a weighted average filter.

        .. note::
            This function requires `Numba` and that the ``data`` has only 1, 2
            or 3 dimensions.

        Parameters
        ----------
        kernel : `numpy.ndarray`, `astropy.convolution.Kernel`
            The kernel (or footprint) for the interpolation. The sum of the
            ``kernel`` must not be 0 (or very close to it). Each axis of the
            kernel must be odd.

        uncertainty : `bool`, optional
            Create a new uncertainty by calculating the variance of the
            filtered values. This at least doubles the total runtime!
            Default is ``False``.

        Returns
        -------
        filtered : `~nddata.nddata.NDDataBase` instance
            Same class as self but new instance containing the convolved data,
            mask (and uncertainty). All other attributes remain the same.

        Examples
        --------

        >>> from nddata.nddata import NDData
        >>> import numpy as np
        >>> ndd = NDData([1,100,1], mask=np.array([0,1,0], dtype=bool))
        >>> ndd2 = ndd.filter_average([1,1,1])
        >>> ndd2
        NDData([ 1.,  1.,  1.])
        >>> ndd2.mask
        array([False, False, False], dtype=bool)

        >>> ndd = NDData([1,2,3,100], mask=np.array([0,0,0,1], dtype=bool))
        >>> ndd2 = ndd.filter_average([1,1,1], True)
        >>> ndd2
        NDData([ 1.5,  2. ,  2.5,  3. ])
        >>> ndd2.uncertainty
        VarianceUncertainty([ 0.25      ,  0.66666667,  0.25      ,  \
0.        ])
        """
        kwargs = {'kernel': kernel, 'rescale': False, 'var': uncertainty}
        return self._filter_convolve(**kwargs)

    def filter_sum(self, kernel, uncertainty=False):
        """Filter the instance data by applying a weighted sum filter.

        .. note::
            This function requires `Numba` and that the ``data`` has only 1, 2
            or 3 dimensions.

        Parameters
        ----------
        kernel : `numpy.ndarray`, `astropy.convolution.Kernel`
            The kernel (or footprint) for the interpolation. The sum of the
            ``kernel`` must not be 0 (or very close to it). Each axis of the
            kernel must be odd.

        uncertainty : `bool`, optional
            Create a new uncertainty by calculating the variance of the
            filtered values. This at least doubles the total runtime!
            Default is ``False``.

        Returns
        -------
        filtered : `~nddata.nddata.NDDataBase` instance
            Same class as self but new instance containing the convolved data,
            mask (and uncertainty). All other attributes remain the same.

        Examples
        --------

        >>> from nddata.nddata import NDData
        >>> import numpy as np

        >>> ndd = NDData([1,2,3,100])
        >>> ndd2 = ndd.filter_median([1,1,1], 'robust')
        >>> ndd2
        NDData([  1.5,   2. ,   3. ,  51.5])
        >>> ndd2.uncertainty
        StdDevUncertainty([  0.        ,   1.48260222,   1.48260222,  \
71.16490649])

        >>> ndd = NDData([1,100,1], mask=np.array([0,1,0], dtype=bool))
        >>> ndd2 = ndd.filter_sum([1,1,1])
        >>> ndd2
        NDData([ 3.,  3.,  3.])
        >>> ndd2.mask
        array([False, False, False], dtype=bool)

        >>> ndd = NDData([1,2,3,100], mask=np.array([0,0,0,1], dtype=bool))
        >>> ndd2 = ndd.filter_sum([1,1,1], True)
        >>> ndd2
        NDData([ 4.5,  6. ,  7.5,  9. ])
        >>> ndd2.uncertainty
        VarianceUncertainty([ 2.25,  6.  ,  2.25,  0.  ])
        """
        kwargs = {'kernel': kernel, 'rescale': True, 'var': uncertainty}
        return self._filter_convolve(**kwargs)

    def filter_median(self, kernel, uncertainty=False):
        """Filter the instance data by applying a median filter.

        .. note::
            This function requires `Numba` and that the ``data`` has only 1, 2
            or 3 dimensions.

        Parameters
        ----------
        kernel : `numpy.ndarray`, `astropy.convolution.Kernel`
            The kernel (or footprint) for the interpolation. Each axis of the
            kernel must be odd.

        uncertainty : `bool` or ``"robust"``, optional
            Create a new uncertainty by calculating the median absolute
            deviation of the filtered values. This at least doubles the total
            runtime! If ``"robust"`` the result is multiplied a correction
            factor of approximatly ``1.428``.
            Default is ``False``.

        Returns
        -------
        filtered : `~nddata.nddata.NDDataBase` instance
            Same class as self but new instance containing the convolved data,
            mask (and uncertainty). All other attributes remain the same.

        Examples
        --------

        >>> from nddata.nddata import NDData
        >>> import numpy as np
        >>> ndd = NDData([1,100,1], mask=np.array([0,1,0], dtype=bool))
        >>> ndd2 = ndd.filter_median([1,1,1])
        >>> ndd2
        NDData([ 1.,  1.,  1.])
        >>> ndd2.mask
        array([False, False, False], dtype=bool)

        >>> ndd = NDData([1,2,3,100], mask=np.array([0,0,0,1], dtype=bool))
        >>> ndd2 = ndd.filter_median([1,1,1], True)
        >>> ndd2
        NDData([ 1.5,  2. ,  2.5,  3. ])
        >>> ndd2.uncertainty
        StdDevUncertainty([ 0.,  1.,  0.,  0.])

        >>> ndd2 = ndd.filter_median([1,1,1], 'robust')
        >>> ndd2.uncertainty
        StdDevUncertainty([ 0.        ,  1.48260222,  0.        ,  0.        ])
        """
        kwargs = self._filter_get_invariants()
        if uncertainty:
            c_data, c_uncert = convolve_median(self.data, kernel,
                                               self._filter_get_mask(),
                                               mad=uncertainty)
            kwargs['uncertainty'] = StdDevUncertainty(c_uncert, copy=False)
        else:
            c_data = convolve_median(self.data, kernel,
                                     self._filter_get_mask(), mad=False)
            kwargs['uncertainty'] = None

        kwargs['mask'] = np.isnan(c_data)
        return self.__class__(c_data, **kwargs)

    def _filter_convolve(self, kernel, rescale, var):
        """Average and sum convolution are the same function so this internal
        method sets it up correctly.
        """
        kwargs = self._filter_get_invariants()
        if var:
            c_data, c_uncert = convolve(self.data, kernel,
                                        self._filter_get_mask(),
                                        rescale=rescale, var=True)
            kwargs['uncertainty'] = VarianceUncertainty(c_uncert,
                                                        copy=False)
        else:
            c_data = convolve(self.data, kernel, self._filter_get_mask(),
                              rescale=rescale, var=False)
            kwargs['uncertainty'] = None

        kwargs['mask'] = np.isnan(c_data)
        return self.__class__(c_data, **kwargs)

    def _filter_get_invariants(self):
        """Attributes that do not change during convolution.
        """
        return {'wcs': self.wcs, 'meta': self.meta, 'unit': self.unit,
                'flags': self.flags}

    def _filter_get_mask(self):
        """
        See also
        --------
        NDStatsMixin._stats_get_mask
        NDReduceMixin._reduce_get_mask
        NDClippingMixin._clipping_get_mask
        NDPlottingMixin._plotting_get_mask
        """
        if isinstance(self.mask, np.ndarray) and self.mask.dtype == bool:
            return self.mask
        # The default is an empty mask with the same shape because we don't
        # just clip the masked values but create a masked array we operate on.
        return np.zeros(self.data.shape, dtype=bool)
        # numpy 1.11 also special cases False and True but not before, so this
        # function is awfully slow then.
        # return False
