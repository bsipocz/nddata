# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from astropy.utils.console import ProgressBar
from traitlets import TraitError


__all__ = ['ProgressBar']


class ProgressBar(ProgressBar):  # pragma: no cover
    """Like `~astropy.utils.console.ProgressBar` but always tries to load \
            the Ipython widget.

    .. note::
        The problem is that I haven't seen the progressbar in notebooks,
        probably some mess-up with stdout or stdin.

    Parameters
    ----------
    total_or_items, file : any type
        See `~astropy.utils.console.ProgressBar` for information.
    """
    def __init__(self, total_or_items, file=None):
        try:
            super(ProgressBar, self).__init__(total_or_items, True, file)
        except TraitError:
            super(ProgressBar, self).__init__(total_or_items, False, file)
