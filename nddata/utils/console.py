# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from astropy.utils.console import ProgressBar

from ..deps import OPT_DEPS


__all__ = ['ProgressBar']


class ProgressBar(ProgressBar):  # pragma: no cover
    """Like `~astropy.utils.console.ProgressBar` but always tries to load \
            the ``Ipython`` widget.

    .. note::
        The problem is that I haven't seen the progressbar in notebooks,
        probably some mess-up with `sys.stdout` or `sys.stderr`. This class
        therefore always tries to use the IPython widget and only if that fails
        uses the text-based one.

    Parameters
    ----------
    total_or_items, file : any type
        See `~astropy.utils.console.ProgressBar` for information.
    """
    def __init__(self, total_or_items, file=None):
        # If traitlets is installed we can try to use the widget for the
        # progressbar (if it fails with a traits error we can fallback to
        # no-widget display)
        if OPT_DEPS['TRAITLETS']:
            from traitlets import TraitError
            try:
                super(ProgressBar, self).__init__(total_or_items, True, file)
            # Possible exceptions:
            # TraitError : IPython is installed but it's not possible to start
            #              the widget.
            # ImportError : IPython widgets are not avaiable (or something else
            #               which would be needed).
            # NameError : I think this is an upstream problem with older
            #             astropy versions which have some problems with the
            #             version determination of IPython...
            # maybe better catch "Exception" here...
            except (TraitError, ImportError, NameError):
                super(ProgressBar, self).__init__(total_or_items, False, file)
        # No traitlets means we must use the no-widget progressbar.
        else:
            super(ProgressBar, self).__init__(total_or_items, False, file)

    def map(self, *args, **kwargs):
        """See :meth:`~astropy.utils.console.ProgressBar.map` for more infos.
        """
        return super(ProgressBar, self).map(*args, **kwargs)

    def update(self, *args, **kwargs):
        """See :meth:`~astropy.utils.console.ProgressBar.update` for more infos.
        """
        return super(ProgressBar, self).update(*args, **kwargs)
