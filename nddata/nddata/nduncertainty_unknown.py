# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from .meta import NDUncertainty

__all__ = ['UnknownUncertainty']


class UnknownUncertainty(NDUncertainty):
    """This implements any unknown uncertainty type.

    The main purpose of having an unknown uncertainty class is to prevent
    uncertainty propagation.

    Parameters
    ----------
    args, kwargs :
        see `~meta.NDUncertainty`
    """

    @property
    def uncertainty_type(self):
        """(``"unknown"``) `UnknownUncertainty` implements any unknown \
                uncertainty type.
        """
        return 'unknown'
