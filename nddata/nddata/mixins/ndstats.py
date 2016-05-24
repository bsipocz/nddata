# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from collections import OrderedDict

import numpy as np

from astropy.table import Table

from ...utils.stats import mode

__all__ = ['NDStatsMixin']


class NDStatsMixin(object):
    """Mixin class to add methods to get statistics about the \
            `nddata.nddata.NDDataBase` class.
    """
    def stats(self):
        """Gives some statistical properties of the data saved in the instance.

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
        return Table(self._stats())

    def _stats(self):
        stats = OrderedDict()
        self._stats_data(stats)
        self._stats_mask(stats)
        self._stats_meta(stats)
        self._stats_wcs(stats)
        self._stats_flags(stats)
        return stats

    def _stats_data(self, stats):
        data = self.data
        # Delete masked values, this will directly convert it to a 1D array
        # if the mask is not appropriate then ravel it.
        size_initial = data.size
        if isinstance(self.mask, np.ndarray) and self.mask.dtype == bool:
            data = data[~self.mask]
        size_masked = data.size
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
        stats['mode'] = [mode(data)[0]]
        stats['std'] = [np.std(data)]
        stats['var'] = [np.var(data)]
        stats['masked'] = [size_initial - size_masked]
        stats['invalid'] = [size_masked - size_valid]
        return data

    def _stats_mask(self, stats):
        pass

    def _stats_meta(self, stats):
        pass

    def _stats_wcs(self, stats):
        pass

    def _stats_flags(self, stats):
        pass
