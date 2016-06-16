# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from ... import OPT_DEPS

if OPT_DEPS['MATPLOTLIB']:  # pragma: no cover
    import matplotlib.pyplot as plt
    if not OPT_DEPS['WCSAXES']:
        __doctest_skip__ = ['NDPlottingMixin.plot_add_wcs_axes',
                            'NDPlottingMixin.plot_add_wcs_subplot',
                            'NDPlottingMixin.plot_add_wcs_labels']
else:  # pragma: no cover
    __doctest_skip__ = ['*']


__all__ = ['NDPlottingMixin']


class NDPlottingMixin(object):
    """Mixin to allow interaction with matplotlib to create plots and images.

    These are meant for interactive use and only provide a subset of possible
    features.
    """
    def plot_add_wcs_axes(self, *args, **kwargs):
        """Create an `matplotlib.axes.Axes` with ``wcs`` projection.

        .. note::
            `wcsaxes` must be installed and the ``wcs`` of the instance must be
            an `~astropy.wcs.WCS` object (or the equivalent from wcsaxes).

        Parameters
        ----------
        args, kwargs :
            Parameter for :func:`matplotlib.pyplot.axes`. The **projection**
            will be set automatically.

        Returns
        -------
        ax : `matplotlib.axes.Axes`
            The created axes.

        Notes
        -----
        This function adds the saved ``wcs`` as **projection** parameter and
        calls :func:`matplotlib.pyplot.axes`.

        See also
        --------
        plot_add_wcs_subplot

        Examples
        --------

        .. plot::
            :include-source:

            >>> from nddata.nddata import NDData
            >>> import numpy as np
            >>> from nddata.utils.numpyutils import mgrid_for_array
            >>> from astropy.modeling.functional_models import Gaussian2D
            >>> from astropy.wcs import WCS
            >>> # Create some data
            >>> data = Gaussian2D(amplitude=100, x_mean=7, x_stddev=20,
            ...                   y_mean=5, y_stddev=40)
            >>> data = data(*mgrid_for_array(np.empty((100, 100))))
            >>> # Setup some basic wcs
            >>> wcs = WCS(naxis=2)
            >>> wcs.wcs.crpix = [-234, 10]
            >>> wcs.wcs.cdelt = [1, 1]
            >>> wcs.wcs.crval = [0, -45]
            >>> wcs.wcs.ctype = ["RA---AIR", "DEC--AIR"]
            >>> wcs.wcs.cunit = ["deg", "deg"]
            >>> # Create an instance and plot it
            >>> ndd = NDData(data, wcs=wcs)
            >>> ax = ndd.plot_add_wcs_axes()               # adds a plot
            >>> ndd.plot_add_wcs_labels()                  # adds ticks, grid
            >>> im = ndd.plot_imshow()                     # plot the image
            >>> cs = ndd.plot_contour(levels=[10, 20, 50]) # plot contours

        .. note::
            `wcsaxes` doesn't support distortions in the ``wcs``.
        """
        kwargs['projection'] = self.wcs
        return plt.axes(*args, **kwargs)

    def plot_add_wcs_subplot(self, *args, **kwargs):
        """Create an `matplotlib.axes.Axes` as subplot with ``wcs`` projection.

        .. note::
            `wcsaxes` must be installed and the ``wcs`` of the instance must be
            an `~astropy.wcs.WCS` object (or the equivalent from wcsaxes).

        Parameters
        ----------
        args, kwargs :
            Parameter for :func:`matplotlib.pyplot.subplot`. The **projection**
            will be set automatically.

        Returns
        -------
        ax : `matplotlib.axes.Axes`
            The created axes.

        Notes
        -----
        This function adds the saved ``wcs`` as **projection** parameter and
        calls :func:`matplotlib.pyplot.subplot`.

        See also
        --------
        plot_add_wcs_axes

        Examples
        --------

        .. plot::
            :include-source:

            >>> from nddata.nddata import NDData
            >>> import numpy as np
            >>> from nddata.utils.numpyutils import mgrid_for_array
            >>> from astropy.modeling.functional_models import AiryDisk2D
            >>> from astropy.wcs import WCS
            >>> # Create some data
            >>> data = AiryDisk2D(amplitude=1000, x_0=40, y_0=40, radius=20)
            >>> data = data(*mgrid_for_array(np.empty((100, 100))))
            >>> # Setup some basic wcs
            >>> wcs = WCS(naxis=2)
            >>> wcs.wcs.crpix = [-234, 10]
            >>> wcs.wcs.cdelt = [1, 1]
            >>> wcs.wcs.crval = [0, -45]
            >>> wcs.wcs.ctype = ["RA---AIR", "DEC--AIR"]
            >>> wcs.wcs.cunit = ["deg", "deg"]
            >>> ndd = NDData(data, wcs=wcs)
            >>> # Split it into two and plot them in one figure:
            >>> ndd1 = ndd[:, :50]
            >>> ndd2 = ndd[:, 50:]
            >>> # Plot left part
            >>> ax1 = ndd1.plot_add_wcs_subplot(1, 2, 1)
            >>> ndd1.plot_add_wcs_labels()
            >>> im = ndd1.plot_imshow()
            >>> # Plot right part
            >>> ax2 = ndd2.plot_add_wcs_subplot(1, 2, 2)
            >>> ndd2.plot_add_wcs_labels()
            >>> im = ndd2.plot_imshow()

        .. note::
            `wcsaxes` doesn't support distortions in the ``wcs``.
        """
        kwargs['projection'] = self.wcs
        return plt.subplot(*args, **kwargs)

    def plot_add_wcs_labels(self, ax=None):
        """Add labels for the axis and display the grid.

        Parameters
        ----------
        ax : `wcsaxes.WCSAxes` or `None`, optional
            If ``None`` operate on the currently active axes otherwise operate
            on the ``axes`` given.
            Default is ``None``.

        Notes
        -----
        A white ``grid`` is enabled and the axis will be labelled according to
        the ``ctype`` of the saved ``wcs``.

        Examples
        --------

        .. plot::
            :include-source:

            >>> from nddata.nddata import NDData
            >>> import numpy as np
            >>> from nddata.utils.numpyutils import mgrid_for_array
            >>> from astropy.modeling.functional_models import AiryDisk2D
            >>> from astropy.wcs import WCS
            >>> # Create some data
            >>> data = AiryDisk2D(amplitude=1000, x_0=40, y_0=40, radius=20)
            >>> data = data(*mgrid_for_array(np.empty((100, 100))))
            >>> # Setup some basic wcs
            >>> wcs = WCS(naxis=2)
            >>> wcs.wcs.crpix = [-234, 10]
            >>> wcs.wcs.cdelt = [1, 1]
            >>> wcs.wcs.crval = [0, -45]
            >>> wcs.wcs.ctype = ["RA---AIR", "DEC--AIR"]
            >>> wcs.wcs.cunit = ["deg", "deg"]
            >>> ndd = NDData(data, wcs=wcs)
            >>> # Plot it two times without (left) and with (right) labels.
            >>> # Left image
            >>> ax1 = ndd.plot_add_wcs_subplot(1, 2, 1)
            >>> im = ndd.plot_imshow()
            >>> cs = ndd.plot_contour(levels=[10])
            >>> # Right image
            >>> ax2 = ndd.plot_add_wcs_subplot(1, 2, 2)
            >>> ndd.plot_add_wcs_labels()
            >>> im = ndd.plot_imshow()
            >>> cs = ndd.plot_contour(levels=[10])
        """
        if ax is None:
            ax = plt.gca()
        ax.coords.grid(color='white')
        ax.coords[0].set_axislabel(self.wcs.wcs.ctype[0])
        ax.coords[1].set_axislabel(self.wcs.wcs.ctype[1])

    def plot_imshow(self, ax=None, **kwargs):
        """:func:`matplotlib.pyplot.imshow` wrapper.

        Parameters
        ----------
        ax : `matplotlib.axes.Axes` or `None`, optional
            If ``None`` operate on the currently active axes otherwise operate
            on the ``axes`` given.
            Default is ``None``.

        kwargs :
            Parameters passed to :func:`matplotlib.pyplot.imshow`.

        Returns
        -------
        imshow : any type
            Whatever the wrapped ``Matplotlib`` function returns.

        Notes
        -----
        This method wraps :func:`matplotlib.pyplot.imshow` by providing some
        fixed parameters:

        - **data** : ``data`` saved in the instance.

        And some default parameters that can be overridden:

        - **origin** : ``"lower"``
        - **interpolation** : ``"none"``
        - **cmap** : ``matplotlib.pyplot.cm.gray``

        Examples
        --------

        .. plot::
            :include-source:

            >>> from nddata.nddata import NDData
            >>> import numpy as np
            >>> from nddata.utils.numpyutils import mgrid_for_array
            >>> from astropy.modeling.functional_models import AiryDisk2D
            >>> import matplotlib.pyplot as plt
            >>> # Create some data
            >>> data = AiryDisk2D(amplitude=1000, x_0=40, y_0=40, radius=20)
            >>> data = data(*mgrid_for_array(np.empty((100, 100))))
            >>> ndd = NDData(data)
            >>> # Create one ndd without mask and one with mask
            >>> ndd2 = ndd.copy()
            >>> mask = np.zeros((100, 100), dtype=bool)
            >>> mask[30:50, 30:50] = 1
            >>> ndd2.mask = mask
            >>> # Create an image with the unmasked (left) data.
            >>> ax1 = plt.subplot(1, 2, 1)
            >>> im = ndd.plot_imshow()
            >>> # and masked (right) data.
            >>> ax2 = plt.subplot(1, 2, 2)
            >>> im = ndd2.plot_imshow()

        .. note::
            If you have a ``mask`` consider convolving the image before
            plotting it to avoid these blank spaces.
        """
        if ax is None:
            ax = plt.gca()

        # Default set of parameters.
        dkwargs = {'origin': 'lower',
                   'interpolation': 'none',
                   'cmap': plt.cm.gray}
        # Update the default parameters by the explicitly given ones. Note that
        # this way the defaults can be overridden. Do not use:
        # kwargs.update(dkwargs) because that would override explicitly given
        # parameters.
        dkwargs.update(kwargs)

        # imshow doesn't support a grid but it can "handle" (not really but at
        # least it tries to do) masked arrays.
        return ax.imshow(self._plotting_get_masked_data(), **dkwargs)

    def plot_contour(self, ax=None, **kwargs):
        """:func:`matplotlib.pyplot.contour` wrapper.

        Parameters
        ----------
        ax : `matplotlib.axes.Axes` or `None`, optional
            If ``None`` operate on the currently active axes otherwise operate
            on the ``axes`` given.
            Default is ``None``.

        Returns
        -------
        contour : any type
            Whatever the wrapped ``Matplotlib`` function returns.

        Notes
        -----
        This method wraps :func:`matplotlib.pyplot.contour` by providing some
        fixed parameters:

        - **data** : ``data`` saved in the instance.

        And some default parameters that can be overridden:

        - **colors** : ``"white"``
        - **alpha** : ``0.5``

        See also
        --------
        nddata.utils.stats.minmax : To provide ``levels`` based on \
            :func:`numpy.linspace` or :func:`numpy.logspace`.

        Examples
        --------

        .. plot::
            :include-source:

            >>> from nddata.nddata import NDData
            >>> import numpy as np
            >>> from nddata.utils.numpyutils import mgrid_for_array
            >>> from astropy.modeling.functional_models import AiryDisk2D
            >>> import matplotlib.pyplot as plt
            >>> # Create some data
            >>> data = AiryDisk2D(amplitude=1000, x_0=40, y_0=40, radius=20)
            >>> data = data(*mgrid_for_array(np.empty((100, 100))))
            >>> ndd = NDData(data)
            >>> # Create one ndd without mask and one with mask
            >>> ndd2 = ndd.copy()
            >>> mask = np.zeros((100, 100), dtype=bool)
            >>> mask[30:50, 30:50] = 1
            >>> ndd2.mask = mask
            >>> # Masks that are no boolean numpy arrays are interpreted as
            >>> # masks containing no masked value. For example "False".
            >>> ndd.mask = False
            >>> # Create a contour with the unmasked (left) data.
            >>> from nddata.utils.stats import minmax
            >>> min_, max_ = minmax(ndd.data)
            >>> logspace = np.linspace(min_, max_, num=10)
            >>> ax1 = plt.subplot(1, 2, 1)
            >>> im = ndd.plot_contour(levels=logspace, colors='black')
            >>> # and masked (right) data
            >>> ax2 = plt.subplot(1, 2, 2)
            >>> im = ndd2.plot_contour(levels=logspace, colors='black')

        .. note::
            If you have a ``mask`` consider convolving the image before
            plotting it to avoid these blank spaces.
        """
        if ax is None:
            ax = plt.gca()

        dkwargs = {'colors': 'white',
                   'alpha': 0.5}
        dkwargs.update(kwargs)

        # Contour takes the data as grid of x, y and z coordinates.
        return ax.contour(self._plotting_get_masked_data(), **dkwargs)

    def plot_hist(self, ax=None, **kwargs):
        """:func:`matplotlib.pyplot.hist` wrapper.

        .. note::
            The data is always ravelled (flattened to one dimension) before the
            histogram is created.

        Parameters
        ----------
        ax : `matplotlib.axes.Axes` or `None`, optional
            If ``None`` operate on the currently active axes otherwise operate
            on the ``axes`` given.
            Default is ``None``.

        Returns
        -------
        histogram : any type
            Whatever the wrapped ``Matplotlib`` function returns.

        Notes
        -----
        This method wraps :func:`matplotlib.pyplot.hist` by providing some
        fixed parameters:

        - **data** : ``data`` saved in the instance.

        And some default parameters that can be overridden:

        - **facecolor** : ``"black"``
        - **alpha** : ``0.5``

        See also
        --------
        nddata.utils.stats.minmax : To provide ``bins`` based on \
            :func:`numpy.arange`.

        Examples
        --------

        .. plot::
            :include-source:

            >>> from nddata.nddata import NDData
            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> # Create some data
            >>> data = np.random.normal(10, 2, (200, 200))
            >>> ndd1 = NDData(data)
            >>> # Create one ndd without mask and one with mask
            >>> ndd2 = ndd1.copy()
            >>> ndd2.clip_sigma(sigma=2)
            >>> # Create an image with the unmasked (left) data.
            >>> from nddata.utils.stats import minmax
            >>> min_, max_ = minmax(ndd1.data)
            >>> bins = np.arange(int(min_)-0.5, int(max_)+1.5, 1)
            >>> ax1 = plt.subplot(1, 2, 1)
            >>> im = ndd1.plot_hist(bins=bins)
            >>> # and masked (right) data
            >>> ax2 = plt.subplot(1, 2, 2)
            >>> im = ndd2.plot_hist(bins=bins)
        """
        if ax is None:
            ax = plt.gca()

        dkwargs = {'facecolor': 'black',
                   'alpha': 0.5}
        dkwargs.update(kwargs)

        data = self._plotting_get_masked_data()
        data = data.data[~data.mask]

        # Contour takes the data as grid of x, y and z coordinates.
        return ax.hist(data, **dkwargs)

    def _plotting_get_mask(self):
        """
        See also
        --------
        NDStatsMixin._stats_get_mask
        NDReduceMixin._reduce_get_mask
        NDClippingMixin._clipping_get_mask
        NDFilterMixin._filter_get_mask
        """
        if isinstance(self.mask, np.ndarray) and self.mask.dtype == bool:
            return self.mask
        # The default is an empty mask with the same shape because we don't
        # just clip the masked values but create a masked array we operate on.
        return np.zeros(self.data.shape, dtype=bool)
        # numpy 1.11 also special cases False and True but not before, so this
        # function is awfully slow then.
        # return False

    def _plotting_get_masked_data(self):
        """Returns a masked array if a mask is present otherwise just returns
        the data.
        """
        return np.ma.array(self.data, mask=self._plotting_get_mask())
