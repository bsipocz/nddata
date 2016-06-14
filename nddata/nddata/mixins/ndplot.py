# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from ... import OPT_DEPS

if OPT_DEPS['MATPLOTLIB']:
    import matplotlib.pyplot as plt


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

        return ax.imshow(self.data, **dkwargs)

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
        """
        if ax is None:
            ax = plt.gca()

        dkwargs = {'colors': 'white',
                   'alpha': 0.5}
        dkwargs.update(kwargs)

        return ax.contour(self.data, **dkwargs)
