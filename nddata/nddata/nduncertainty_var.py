# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from astropy.units import Quantity
import astropy.units as u

from .meta import NDUncertaintyGaussian

__all__ = ['VarianceUncertainty']


class VarianceUncertainty(NDUncertaintyGaussian):
    """Variance uncertainty assuming first order gaussian error \
            propagation.

    This class implements uncertainty propagation for ``addition``,
    ``subtraction``, ``multiplication`` and ``division``. The class can handle
    if the uncertainty has a unit that differs from (but is convertible to) the
    parents `NDData` unit. The unit of the resulting uncertainty will have the
    same unit as the resulting data. Also support for correlation is possible
    but requires the correlation as input. It cannot handle correlation
    determination itself.

    Parameters
    ----------
    args, kwargs :
        see `~meta.NDUncertainty`

    Examples
    --------
    `VarianceUncertainty` should always be associated with an `NDDataBase`-like
    instance, either by creating it during initialization::

        >>> from nddata.nddata import NDData, VarianceUncertainty
        >>> ndd = NDData([1,2,3],
        ...              uncertainty=VarianceUncertainty([0.1, 0.1, 0.1]))
        >>> ndd.uncertainty
        VarianceUncertainty([ 0.1,  0.1,  0.1])

    or by setting it manually on a `NDData` instance::

        >>> ndd.uncertainty = VarianceUncertainty([0.2], unit='m', copy=True)
        >>> ndd.uncertainty
        VarianceUncertainty([ 0.2])

    the uncertainty ``data`` can also be set directly::

        >>> ndd.uncertainty.data = 2
        >>> ndd.uncertainty
        VarianceUncertainty(2)
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
    def effective_unit(self):
        """(`~astropy.units.Unit`) The effective unit of the instance is the \
            ``unit`` of the uncertainty or, if not set, the squared unit of \
            the parent.
        """
        if self.unit is None:
            # The uncertainty has no unit by itself, check if the parent has a
            # unit and return it. VarianceUncertainty should have the squared
            # dimension as the data so square it if necessary. If it has
            # no parent let the MissingDataAssociationException bubble up, we
            # would expect to find a unit if this property is accessed.
            if self.parent_nddata.unit is not None:
                return self.parent_nddata.unit ** 2
            else:
                return None
        return self._unit

    @property
    def supports_correlated(self):
        """(`True`) `VarianceUncertainty` allows to propagate correlated \
                      uncertainties.

        ``correlation`` must be given, this class does not implement computing
        it by itself.
        """
        return True

    @property
    def uncertainty_type(self):
        """(``"var"``) `VarianceUncertainty` implements variance deviation.
        """
        return 'var'

    def convert_unit_to(self, unit, equivalencies=[]):
        """Returns an uncertainty converted to the new unit.

        This method requires that the `effective_unit` is not ``None``.

        See also :meth:`~nddata.nddata.mixins.NDUnitConvMixin.convert_unit_to`.

        .. warning::
            VarianceUncertainty will convert to the squared given unit.
        """
        # convert it explicitly to a Unit object so that we can square it.
        unit = u.Unit(unit) ** 2
        conv_data = self.effective_unit.to(unit, self.data, equivalencies)
        return self.__class__(conv_data, unit, copy=False)

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

        # Check if the unit of the resulting uncertainty is identical to the
        # on of the result (squared) and if so drop it.
        if isinstance(result, u.Quantity):
            if isinstance(result_data, u.Quantity):
                # Both are Quantities - we can drop the unit if they have the
                # same unit.
                if result.unit == result_data.unit ** 2:
                    result = result.value
            else:
                # Only the uncertainty is a Quantity - we can drop the unit of
                # the uncertainty if it's dimensionless.
                if result.unit == u.dimensionless_unscaled:
                    result = result.value

        return self.__class__(result, copy=False)

    def _propagate_add(self, other_uncert, result_data, correlation):

        # Formula: sigma = dA + dB - 2 * cor * sqrt(dA*dB)

        # Any uncertainty with None is considered 0
        dA = 0 if self.data is None else self.data
        dB = 0 if other_uncert.data is None else other_uncert.data

        # Then apply the units if necessary
        if self.effective_unit is not None:
            dA = dA * self.effective_unit
        if other_uncert.effective_unit is not None:
            dB = dB * other_uncert.effective_unit

        # Calculate the result including correlation if necessary
        if isinstance(correlation, np.ndarray) or correlation != 0:
            result = dA + dB + 2 * correlation * np.sqrt(dA * dB)
        else:
            result = dA + dB

        return result

    def _propagate_subtract(self, other_uncert, result_data, correlation):

        # Formula: sigma = dA + dB - 2 * cor * sqrt(dA*dB)

        # Same as addition but subtracting the correlation term.

        # Any uncertainty with None is considered 0
        dA = 0 if self.data is None else self.data
        dB = 0 if other_uncert.data is None else other_uncert.data

        # Then apply the units if necessary
        if self.effective_unit is not None:
            dA = dA * self.effective_unit
        if other_uncert.effective_unit is not None:
            dB = dB * other_uncert.effective_unit

        # Calculate the result including correlation if necessary
        if isinstance(correlation, np.ndarray) or correlation != 0:
            result = dA + dB - 2 * correlation * np.sqrt(dA * dB)
        else:
            result = dA + dB

        return result

    def _propagate_multiply(self, other_uncert, result_data, correlation):

        # Formula: sigma = dA*B**2 + dB*A**2 + 2 * rho * A * B * sqrt(dA dB)

        # Any uncertainty or parent with None is considered 0
        A = 0 if self.parent_nddata.data is None else self.parent_nddata.data
        dA = 0 if self.data is None else self.data
        B = (0 if other_uncert.parent_nddata.data is None
             else other_uncert.parent_nddata.data)
        dB = 0 if other_uncert.data is None else other_uncert.data

        # Then apply the units if necessary.
        if self.parent_nddata.unit is not None:
            A = A * self.parent_nddata.unit
        if self.effective_unit is not None:
            dA = dA * self.effective_unit
        if other_uncert.parent_nddata.unit is not None:
            B = B * other_uncert.parent_nddata.unit
            parent_unit_squared = other_uncert.parent_nddata.unit ** 2
            if parent_unit_squared != other_uncert.effective_unit:
                B = B.to(other_uncert.effective_unit ** (1/2))
        if other_uncert.effective_unit is not None:
            dB = dB * other_uncert.effective_unit

        # Calculate some intermediate values so they will not computed twice
        # in case correlation is given.
        dAB2 = dA * B ** 2
        dBA2 = dB * A ** 2

        # Calculate the result with or without correlation
        if isinstance(correlation, np.ndarray) or correlation != 0:
            result = dAB2 + dBA2 + 2 * correlation * np.sqrt(dAB2 * dBA2)
        else:
            result = dAB2 + dBA2

        return result

    def _propagate_divide(self, other_uncert, result_data, correlation):

        # Formula: sigma = dA/B**2 + dBA**2/B**4 - 2 rho A / B**3 * sqrt(dA dB)

        # Any uncertainty or parent with None is considered 0
        A = 0 if self.parent_nddata.data is None else self.parent_nddata.data
        dA = 0 if self.data is None else self.data
        B = (0 if other_uncert.parent_nddata.data is None
             else other_uncert.parent_nddata.data)
        dB = 0 if other_uncert.data is None else other_uncert.data

        # Then apply the units if necessary.
        if self.parent_nddata.unit is not None:
            A = A * self.parent_nddata.unit
        if self.effective_unit is not None:
            dA = dA * self.effective_unit
        if other_uncert.parent_nddata.unit is not None:
            B = B * other_uncert.parent_nddata.unit
            parent_unit_squared = other_uncert.parent_nddata.unit ** 2
            if parent_unit_squared != other_uncert.effective_unit:
                B = B.to(other_uncert.effective_unit ** (1/2))
        if other_uncert.effective_unit is not None:
            dB = dB * other_uncert.effective_unit

        # Calculate some intermediate values so they will not computed twice
        # in case correlation is given.
        dA_B2 = dA / B ** 2
        A2dB_B4 = dB * A ** 2 / B ** 4

        # Calculate the result with or without correlation
        if isinstance(correlation, np.ndarray) or correlation != 0:
            result = dA_B2 + A2dB_B4 - 2*correlation*np.sqrt(dA_B2*A2dB_B4)
        else:
            result = dA_B2 + A2dB_B4

        return result

    def _propagate_power(self, other_uncert, result_data, correlation):
        # Formula:
        # sigma = dA(B*A**(B-1))**2 + dB*(ln(A)A**B)**2 +
        #              2ln(A)B*A**(2B-1)*rho*sqrt(dAdB)

        # Any uncertainty or parent with None is considered 0
        A = (0 if self.parent_nddata.data is None
             else self.parent_nddata.data)
        dA = 0 if self.data is None else self.data
        B = (0 if other_uncert.parent_nddata.data is None
             else other_uncert.parent_nddata.data)
        dB = np.array(0) if other_uncert.data is None else other_uncert.data

        # dB is a numpy array so we can find out if something is a
        # scalar by checking how many elements it contains.
        # exponent_scalar = B.size == 1
        exponent_uncertainty = dB.size > 1 or dB != 0

        # Then apply the units if necessary.

        # Power is a bit special with units. But it boils down to:
        # 1) Exponent must be dimensionless or convertible to dimensionless
        # 2) Base can only hava a unit if the exponent is a scalar and has no
        #    uncertainty.

        # I also calculate the result of np.log(A) already in here because in
        # case the exponent has no uncertainty - it can be immediatly set to 0.
        if self.parent_nddata.unit is not None:
            # See 2)
            # A must have a unit, so we do not lose the connection to the unit
            # of the result of the parent. Cost me a lot of time finding THAT
            # out ... the hard way ;-)
            A = A * self.parent_nddata.unit
            if exponent_uncertainty:
                A = A.to(u.dimensionless_unscaled)
                lnA = np.log(A)
            else:
                # Here we again convert the unit of A so that the final result
                # will have the expected unit. This is not necessary in the
                # "if" part!
                if self.parent_nddata.unit ** 2 != self.effective_unit:
                    A = A.to(self.effective_unit ** (1/2))
                lnA = 0
        else:
            lnA = np.log(A)

        if self.effective_unit is not None:
            # see 2)
            if exponent_uncertainty:
                dA = self.effective_unit.to(u.dimensionless_unscaled, dA)
            else:
                dA = dA * self.effective_unit

        if other_uncert.parent_nddata.unit is not None:
            # see 1)
            B = other_uncert.parent_nddata.unit.to(u.dimensionless_unscaled, B)

        if other_uncert.effective_unit is not None:
            # see 1)
            dB = other_uncert.effective_unit.to(u.dimensionless_unscaled, dB)

        # Calculate some intermediate values so they will not computed twice
        # in case correlation is given.
        dBlnAA_B = dB * (A ** B * lnA) ** 2

        BdAA_Bm1 = dA * (B * A ** (B - 1)) ** 2

        # Calculate the result with or without correlation
        if isinstance(correlation, np.ndarray) or correlation != 0:
            result = (dBlnAA_B + BdAA_Bm1 + 2 * correlation *
                      np.sqrt(dBlnAA_B * BdAA_Bm1))
        else:
            result = dBlnAA_B + BdAA_Bm1

        return result
