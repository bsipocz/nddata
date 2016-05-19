# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__all__ = ['MissingDataAssociationException',
           'IncompatibleUncertaintiesException']


class IncompatibleUncertaintiesException(Exception):
    """This exception is raised if different instances of \
            `~.meta.NDUncertainty` are incompatible.
    """


class MissingDataAssociationException(Exception):
    """This exception is raised if the parent of `~.meta.NDUncertainty` is not\
            set but accessed.
    """
