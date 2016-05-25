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

__all__ = ['NDStatsMixin']


class NDStatsMixin(object):
    """Mixin class to add methods to get statistics about the \
            `nddata.nddata.NDDataBase` class.
    """
    def stats(self, scipy=False, astropy=False, decimals_mode=0):
        """Gives some statistical properties of the data saved in the instance.

        .. note::
            If the ``mask`` should be taken into account it is needed to be
            a `numpy.ndarray` with a boolean dtype. Otherwise the mask is
            ignored.

        Parameters
        ----------
        scipy : `bool`, optional
            If ``True`` the :func:`scipy.stats.skew` and
            :func:`scipy.stats.kurtosis` are included in the returned table.
            Default is ``False``.

        astropy : `bool`, optional
            If ``True`` the median absolute deviation
            (:func:`astropy.stats.mad_std`), and biweight statistics (
            :func:`astropy.stats.biweight_location` and
            :func:`astropy.stats.biweight_midvariance`) are included in the
            returned table.
            Default is ``False``

        decimals_mode : `int`, optional
            The number of digits relevant **only** for calculating the
            **mode**. See also the parameter description of the function:
            :func:`~nddata.utils.stats.mode`.
            Default is ``0``.

        Returns
        -------
        stats : `~astropy.table.Table`
            The statistical results as table object. Including some information
            on the number of elements and excluded elements:

            - **elements** : number of elements included in the statistics.
            - **masked** : number of elements excluded because they are masked.
            - **invalid** : number of elements excluded because they are \
                considered invalid. Like ``NaN`` or ``Inf``.

            and the results of the statistical computations:

            - **min** : :func:`numpy.amin`
            - **max** : :func:`numpy.amax`
            - **mean** : :func:`numpy.mean`
            - **median** : :func:`numpy.median`
            - **mode** : :func:`~nddata.utils.stats.mode` the mode of the \
                ``data`` rounded to the nearest integer.
            - **std** : :func:`numpy.std`
            - **var** : :func:`numpy.var`

            optional returned results:

            - **skew** : :func:`scipy.stats.skew`
            - **kurtosis** : :func:`scipy.stats.kurtosis`

            - **mad** : :func:`astropy.stats.mad_std`
            - **biweight_location** : :func:`astropy.stats.biweight_location`
            - **biweight_midvariance** : \
                :func:`astropy.stats.biweight_midvariance`

        Examples
        --------
        This mixin is already included in `~nddata.nddata.NDData`::

            >>> from nddata.nddata import NDData
            >>> ndd1 = NDData([1.0,2,3,4,5])
            >>> stats1 = ndd1.stats()
            >>> print(stats1)
            elements min max mean median mode      std      var masked invalid
            -------- --- --- ---- ------ ---- ------------- --- ------ -------
                   5 1.0 5.0  3.0    3.0  1.0 1.41421356237 2.0      0       0

        And it also takes a mask into account::

            >>> import numpy as np
            >>> mask=np.array([True, False, False, False, False])
            >>> ndd2 = NDData(ndd1, mask=mask)
            >>> stats2 = ndd2.stats()
            >>> print(stats2)
            elements min max mean median mode      std      var  masked invalid
            -------- --- --- ---- ------ ---- ------------- ---- ------ -------
                   4 2.0 5.0  3.5    3.5  2.0 1.11803398875 1.25      1       0

        Invalid elements like ``NaN`` or ``Inf`` are removed as well::

            >>> ndd3 = NDData(ndd2.data.astype(float))
            >>> ndd3.data[4] = np.inf
            >>> stats3 = ndd3.stats()
            >>> print(stats3)
            elements min max mean median mode      std      var  masked invalid
            -------- --- --- ---- ------ ---- ------------- ---- ------ -------
                   4 1.0 4.0  2.5    2.5  1.0 1.11803398875 1.25      0       1

        the returned tables are :class:`~astropy.table.Table` instances so they
        can you can treat them as such. For example stack them::

            >>> from astropy.table import vstack
            >>> print(vstack([stats1, stats2, stats3]))
            elements min max mean median mode      std      var  masked invalid
            -------- --- --- ---- ------ ---- ------------- ---- ------ -------
                   5 1.0 5.0  3.0    3.0  1.0 1.41421356237  2.0      0       0
                   4 2.0 5.0  3.5    3.5  2.0 1.11803398875 1.25      1       0
                   4 1.0 4.0  2.5    2.5  1.0 1.11803398875 1.25      0       1
        """
        if self.data is None:
            raise TypeError('cannot do statistics on the data if the data is '
                            'None.')
        return Table(self._stats(scipy=scipy, astropy=astropy,
                                 decimals_mode=decimals_mode))

    def _stats(self, scipy, astropy, decimals_mode):
        """If someone wants to include some more attributes that contribute
        to the returned statistics one can add a function here.
        """
        # Create the ordered dict that will be converted to the table later.
        stats = OrderedDict()
        # Get the mask, the standard is just checking if it's a boolean array
        # and returning it or if it's something else it returns None. But
        # subclasses can modify this method to also evaluate other masks.
        mask = self._stats_get_mask()
        # No need to return anything here, the statistics are inserted in the
        # dictionary and are updated in-place
        self._stats_data(stats, mask, scipy=scipy, astropy=astropy,
                         decimals_mode=decimals_mode)
        return stats

    def _stats_data(self, stats, mask, scipy, astropy, decimals_mode):
        data = self.data

        # The original data size, for computation of valid elements and how
        # many are masked/invalid.
        size_initial = data.size

        # Delete masked values, this will directly convert it to a 1D array
        # if the mask is not appropriate then ravel it.
        if mask is not None:
            data = data[~self.mask]
        size_masked = data.size

        # Delete invalid (NaN, Inf) values. This should ensure that the result
        # is always a 1D array
        data = data[np.isfinite(data)]
        size_valid = data.size
        stats['elements'] = [size_valid]

        stats['min'] = [np.amin(data)]
        stats['max'] = [np.amax(data)]
        stats['mean'] = [np.mean(data)]
        stats['median'] = [np.median(data)]
        # Use custom mode defined in this package because scipy.stats.mode is
        # very, very slow and by default tries to calculate the mode along
        # axis=0 and not for the whole array.
        # Take the first element since the second is the number of occurences.
        stats['mode'] = [mode(data, decimals=decimals_mode)[0]]

        if astropy:
            stats['biweight_location'] = [biweight_location(data)]

        stats['std'] = [np.std(data)]

        if astropy:
            stats['mad'] = [mad_std(data)]
            stats['biweight_midvariance'] = [biweight_midvariance(data)]

        stats['var'] = [np.var(data)]

        if scipy:  # pragma: no cover
            if not OPT_DEPS['SCIPY']:
                log.info('SciPy is not installed.')
            else:
                # Passing axis=None should not be important since we already
                # boolean indexed the array and it's 1D. But it's important
                # to remember that there default is axis=0 and not axis=None!
                stats['skew'] = [skew(data, axis=None)]
                stats['kurtosis'] = [kurtosis(data, axis=None)]

        stats['masked'] = [size_initial - size_masked]
        stats['invalid'] = [size_masked - size_valid]

        return data

    def _stats_get_mask(self):
        """Mostly for subclasses that don't use numpy bool masks as "mask".
        These only need to override this method and evaluate their mask here
        so it can be applied to the "data".
        """
        if isinstance(self.mask, np.ndarray) and self.mask.dtype == bool:
            return self.mask
        return None
