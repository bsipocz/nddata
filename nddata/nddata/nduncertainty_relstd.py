# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

import astropy.units as u

from .meta import NDUncertaintyGaussian

__all__ = ['RelativeUncertainty']


class RelativeUncertainty(NDUncertaintyGaussian):
    """Relative uncertainty assuming first order gaussian error \
            propagation.

    This class implements uncertainty propagation for ``addition``,
    ``subtraction``, ``multiplication`` and ``division``. The class can handle
    if the uncertainty has a unit that differs from (but is convertible to) the
    parents `NDData` unit. Also support for correlation is possible but
    requires the correlation as input. It cannot handle correlation
    determination itself.

    Parameters
    ----------
    args, kwargs :
        see `~meta.NDUncertainty`

    Examples
    --------
    `RelativeUncertainty` should always be associated with an `NDDataBase`-like
    instance, either by creating it during initialization::

        >>> from nddata.nddata import NDData, RelativeUncertainty
        >>> ndd = NDData([1,2,3],
        ...              uncertainty=RelativeUncertainty([0.1, 0.1, 0.1]))
        >>> ndd.uncertainty
        RelativeUncertainty([ 0.1,  0.1,  0.1])

    or by setting it manually on a `NDData` instance::

        >>> ndd.uncertainty = RelativeUncertainty([0.2], unit='', copy=True)
        >>> ndd.uncertainty
        RelativeUncertainty([ 0.2])

    the uncertainty ``data`` can also be set directly::

        >>> ndd.uncertainty.data = 2
        >>> ndd.uncertainty
        RelativeUncertainty(2)
    """
    # propagation methods for one operand operations
    # TODO: Currently no one operand operations are implemented... :-(
    _propagate_1 = {}
    # propagation methods for two operand operations
    _propagate_2 = {np.add: '_propagate_add',
                    np.subtract: '_propagate_subtract',
                    np.multiply: '_propagate_multiply',
                    np.divide: '_propagate_divide',
                    np.true_divide: '_propagate_divide',
                    np.power: '_propagate_power'}

    @property
    def unit(self):
        """(`None`) relative uncertainties have no unit.
        """
        return None

    @unit.setter
    def unit(self, value):
        if value is None:
            pass
        elif value == u.dimensionless_unscaled:
            pass
        else:
            raise TypeError('relative uncertainties don\' have units.')
        self._unit = None

    @property
    def effective_unit(self):
        """(`~astropy.units.Unit`) relative uncertainties have no unit.
        """
        if self.unit is not None:
            raise ValueError('relative uncertainties can\'t have units.')
        return None

    @property
    def supports_correlated(self):
        """(`True`) `RelativeUncertainty` allows to propagate correlated \
                      uncertainties.

        ``correlation`` must be given, this class does not implement computing
        it by itself.
        """
        return True

    @property
    def uncertainty_type(self):
        """(``"rel_std"``) `RelativeUncertainty` implements relative standard \
                deviation.
        """
        return 'rel_std'

    def propagate(self, operation, other_nddata, result_data, correlation):
        """Calculate the resulting uncertainty given an operation on the data.

        Parameters
        ----------
        operation : callable
            The operation that is performed on the `NDData`. Supported are
            `numpy.add`, `numpy.subtract`, `numpy.multiply` and
            `numpy.true_divide` (or `numpy.divide`).

        other_nddata : `NDData`
            The second operand in the arithmetic operation.

        result_data : `~astropy.units.Quantity` or `numpy.ndarray`
            The result of the arithmetic operations on the data.

        correlation : `numpy.ndarray` or number
            The correlation (rho) is defined between the uncertainties in
            sigma_AB = sigma_A * sigma_B * rho. A value of ``0`` means
            uncorrelated operands.

        Returns
        -------
        resulting_uncertainty : `~meta.NDUncertainty` instance
            Another instance of the same `~meta.NDUncertainty` subclass
            containing the uncertainty of the result.

        Raises
        ------
        ValueError
            If the ``operation`` is not supported or if correlation is not zero
            but the subclass does not support correlated uncertainties.

        Notes
        -----
        First this method checks if a correlation is given and the subclass
        implements propagation with correlated uncertainties.
        Then the second uncertainty is converted (or an Exception is raised)
        to the same class in order to do the propagation.
        Then the appropriate propagation method is invoked and the result is
        returned.
        """
        # Check if the subclass supports correlation
        if not self.supports_correlated:
            # If the correlation is not zero or even a numpy array the user
            # specified correlated propagation but the class doesn't support it
            # raise an Exception here
            if isinstance(correlation, np.ndarray) or correlation != 0:
                raise ValueError("{0} does not support uncertainty propagation"
                                 " with correlation."
                                 "".format(self.__class__.__name__))

        # Get the other uncertainty (and convert it to a matching one) using
        # the converter-registry through "from_uncertainty"
        other_uncert = self.from_uncertainty(other_nddata.uncertainty)

        # search if the operation was registered in the propagation dictionary
        # for 2 operand propagation.
        method_name = self._propagate_2.get(operation, None)
        if method_name is not None:
            # The operation was registered so we can simply call it.
            result = getattr(self, method_name)(other_uncert, result_data,
                                                correlation)
        else:
            raise ValueError('unsupported operation')

        # Other uncertainties check if they can drop the unit for relative
        # uncertainty this must be done ... maybe.
        # Because we can only accept dimensionless relative uncertainties.
        if isinstance(result, u.Quantity):
            if result.unit != u.dimensionless_unscaled:
                # It's a Quantity with a unit. Convert it to dimensionless so
                # it will be accepted as result.
                result = result.to(u.dimensionless_unscaled).value

        return self.__class__(result, copy=False)

    def _propagate_add(self, other_uncert, result_data, correlation):
        pass

    def _propagate_subtract(self, other_uncert, result_data, correlation):
        pass

    def _propagate_multiply(self, other_uncert, result_data, correlation):
        pass

    def _propagate_divide(self, other_uncert, result_data, correlation):
        pass

    def _propagate_power(self, other_uncert, result_data, correlation):
        pass
