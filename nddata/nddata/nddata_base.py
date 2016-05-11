# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from abc import ABCMeta, abstractproperty, abstractmethod

from astropy.extern import six


__all__ = ['NDDataBase']


@six.add_metaclass(ABCMeta)
class NDDataBase(object):
    """Base metaclass that defines the interface for N-dimensional datasets with
    associated meta informations used in ``astropy``.

    All properties and ``__init__`` have to be overridden in subclasses. See
    `NDData` for a subclass that defines this interface on `numpy.ndarray`-like
    ``data``.
    """
    @abstractmethod
    def __init__(self):
        pass

    @abstractproperty
    def data(self):
        """The stored dataset.
        """

    @abstractproperty
    def mask(self):
        """Mask for the dataset.

        Masks should follow the ``numpy`` convention that **valid** data points
        are marked by ``False`` and **invalid** ones with ``True``.
        """

    @abstractproperty
    def unit(self):
        """Unit for the dataset.
        """

    @abstractproperty
    def wcs(self):
        """World coordinate system (WCS) for the dataset.
        """

    @abstractproperty
    def meta(self):
        """Additional meta informations about the dataset.

        Should be `dict`-like.
        """

    @abstractproperty
    def uncertainty(self):
        """Uncertainty in the dataset.

        Should have an attribute ``uncertainty_type`` that defines what kind of
        uncertainty is stored, such as ``"std"`` for standard deviation or
        ``"var"`` for variance.
        """
