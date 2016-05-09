# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__all__ = ['MissingParentNDDataException']


class MissingParentNDDataException(Exception):
    """Exception that indicates that one tried to propagate an uncertainty
    which has no reference to the class containing the data.
    """
    pass
