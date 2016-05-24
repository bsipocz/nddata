# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from abc import ABCMeta, abstractproperty, abstractmethod

from astropy.extern import six


__all__ = ['NDDataMeta']


@six.add_metaclass(ABCMeta)
class NDDataMeta(object):
    """Base metaclass (`abc.ABCMeta`) that defines the interface for \
            N-dimensional datasets.

    See also
    --------
    .NDDataBase
    .NDData
    """
    @abstractmethod
    def __init__(self):
        """`abc.abstractmethod` : ``__init__``

        Create an instance by setting the attributes.
        """
        pass

    @abstractproperty
    def data(self):
        """(`abc.abstractproperty`) The stored dataset.
        """

    @abstractproperty
    def mask(self):
        """(`abc.abstractproperty`) Mask for the dataset.
        """

    @abstractproperty
    def unit(self):
        """(`abc.abstractproperty`) Unit for the dataset.
        """

    @abstractproperty
    def wcs(self):
        """(`abc.abstractproperty`) World coordinate system (WCS) for the \
                dataset.
        """

    @abstractproperty
    def meta(self):
        """(`abc.abstractproperty`) Additional meta information about the \
                dataset.
        """

    @abstractproperty
    def uncertainty(self):
        """(`abc.abstractproperty`) Uncertainty in the dataset.
        """

    @abstractproperty
    def flags(self):
        """(`abc.abstractproperty`) Flags for the dataset.
        """
