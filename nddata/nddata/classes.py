# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from abc import ABCMeta, abstractproperty, abstractmethod

import six

__all__ = ['NDDataMeta', 'NDUncertaintyMeta', 'NDUncertaintyPropagationMeta']


@six.add_metaclass(ABCMeta)
class NDDataMeta(object):
    """Metaclass (`abc.ABCMeta`) that defines the container interface for
    N-dimensional datasets.
    """
    @abstractmethod
    def __init__(self):
        """`abc.abstractmethod` : ``__init__``

            Create an instance by setting the attributes.
        """
        pass

    @abstractproperty
    def data(self):
        """`abc.abstractproperty` : the saved data.
        """
        return None

    @abstractproperty
    def mask(self):
        """`abc.abstractproperty` : the mask for the data.
        """
        return None

    @abstractproperty
    def unit(self):
        """`abc.abstractproperty` : the unit for the data.
        """
        return None

    @abstractproperty
    def wcs(self):
        """`abc.abstractproperty` : the world coordinate system (WCS) for the data.
        """
        return None

    @abstractproperty
    def meta(self):
        """`abc.abstractproperty` : additional meta information about the data.
        """
        return None

    @abstractproperty
    def uncertainty(self):
        """`abc.abstractproperty` : the uncertainty in the data.
        """
        return None


@six.add_metaclass(ABCMeta)
class NDUncertaintyMeta(object):
    """Metaclass (`abc.ABCMeta`) that defines the container interface for
    uncertainties in N-dimensional datasets.
    """
    @abstractmethod
    def __init__(self):
        """`abc.abstractmethod` : ``__init__``

            Create an instance by setting the attributes.
        """
        pass

    @abstractproperty
    def data(self):
        """`abc.abstractproperty` : the saved uncertainty.
        """
        return None

    @abstractproperty
    def unit(self):
        """`abc.abstractproperty` : the unit for the uncertainty.
        """
        return None

    @abstractproperty
    def uncertainty_type(self):
        """`abc.abstractproperty` : the kind of uncertainty.
        """
        return None


@six.add_metaclass(ABCMeta)
class NDUncertaintyPropagationMeta(object):
    """Metaclass (`abc.ABCMeta`) that defines the interface for
    propagation of uncertainties.
    """

    @abstractproperty
    def parent_nddata(self):
        """`abc.abstractproperty` : the associated data.
        """
        return None

    @abstractproperty
    def supports_correlated(self):
        """`abc.abstractproperty` : indicator if class supports error \
        propagation with correlated uncertainties.
        """
        return None

    @abstractmethod
    def propagate(self, operation, other, result=None, correlation=None):
        """`abc.abstractmethod` : calculate the resulting uncertainty.
        """
        return None
