# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from copy import deepcopy

import numpy as np

from astropy.units import Quantity

from .meta import NDUncertainty, NDUncertaintyGaussian

__all__ = ['StdDevUncertainty', 'UnknownUncertainty']


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


class StdDevUncertainty(NDUncertaintyGaussian):
    """Standard deviation uncertainty assuming first order gaussian error
    propagation.

    This class implements uncertainty propagation for ``addition``,
    ``subtraction``, ``multiplication`` and ``division`` with other instances
    of `StdDevUncertainty`. The class can handle if the uncertainty has a
    unit that differs from (but is convertible to) the parents `NDData` unit.
    The unit of the resulting uncertainty will have the same unit as the
    resulting data. Also support for correlation is possible but requires the
    correlation as input. It cannot handle correlation determination itself.

    Parameters
    ----------
    args, kwargs :
        see `~meta.NDUncertainty`

    Examples
    --------
    `StdDevUncertainty` should always be associated with an `NDDataBase`-like
    instance, either by creating it during initialization::

        >>> from nddata.nddata import NDData, StdDevUncertainty
        >>> ndd = NDData([1,2,3],
        ...              uncertainty=StdDevUncertainty([0.1, 0.1, 0.1]))
        >>> ndd.uncertainty
        StdDevUncertainty([ 0.1,  0.1,  0.1])

    or by setting it manually on a `NDData` instance::

        >>> ndd.uncertainty = StdDevUncertainty([0.2], unit='m', copy=True)
        >>> ndd.uncertainty
        StdDevUncertainty([ 0.2])

    the uncertainty ``array`` can also be set directly::

        >>> ndd.uncertainty.array = 2
        >>> ndd.uncertainty
        StdDevUncertainty(2)
    """

    @property
    def supports_correlated(self):
        """(`True`) `StdDevUncertainty` allows to propagate correlated \
                      uncertainties.

        ``correlation`` must be given, this class does not implement computing
        it by itself.
        """
        return True

    @property
    def uncertainty_type(self):
        """(``"std"``) `StdDevUncertainty` implements standard deviation.
        """
        return 'std'

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
            if isinstance(correlation, np.ndarray) or correlation != 0:
                raise ValueError("{0} does not support uncertainty propagation"
                                 " with correlation."
                                 "".format(self.__class__.__name__))

        # Get the other uncertainty (and convert it to a matching one)
        other_uncert = self.from_uncertainty(other_nddata.uncertainty)

        if operation.__name__ == 'add':
            result = self._propagate_add(other_uncert, result_data,
                                         correlation)
        elif operation.__name__ == 'subtract':
            result = self._propagate_subtract(other_uncert, result_data,
                                              correlation)
        elif operation.__name__ == 'multiply':
            result = self._propagate_multiply(other_uncert, result_data,
                                              correlation)
        elif operation.__name__ in ['true_divide', 'divide']:
            result = self._propagate_divide(other_uncert, result_data,
                                            correlation)
        else:
            raise ValueError('unsupported operation')

        return self.__class__(result, copy=False)

    def _propagate_add(self, other_uncert, result_data, correlation):

        if self.array is None:
            # Formula: sigma = dB

            if other_uncert.effective_unit is not None and (
                        result_data.unit != other_uncert.effective_unit):
                # If the other uncertainty has a unit and this unit differs
                # from the unit of the result convert it to the results unit
                return (other_uncert.array * other_uncert.unit).to(
                            result_data.unit).value
            else:
                # Copy the result because _propagate will not copy it but for
                # arithmetic operations users will expect copys.
                return deepcopy(other_uncert.array)

        elif other_uncert.array is None:
            # Formula: sigma = dA

            if self.effective_unit is not None and self.effective_unit != self.parent_nddata.unit:
                # If the uncertainty has a different unit than the result we
                # need to convert it to the results unit.
                return (self.array * self.unit).to(result_data.unit).value
            else:
                # Copy the result because _propagate will not copy it but for
                # arithmetic operations users will expect copys.
                return deepcopy(self.array)

        else:
            # Formula: sigma = sqrt(dA**2 + dB**2 + 2*cor*dA*dB)

            # Calculate: dA (this) and dB (other)
            if self.effective_unit != other_uncert.effective_unit:
                # In case the two uncertainties (or data) have different units
                # we need to use quantity operations. The case where only one
                # has a unit and the other doesn't is not possible with
                # addition and would have raised an exception in the data
                # computation
                this = self.array * self.effective_unit
                other = other_uncert.array * other_uncert.effective_unit
            else:
                # Since both units are the same or None we can just use
                # numpy operations
                this = self.array
                other = other_uncert.array

            # Determine the result depending on the correlation
            if isinstance(correlation, np.ndarray) or correlation != 0:
                corr = 2 * correlation * this * other
                result = np.sqrt(this**2 + other**2 + corr)
            else:
                result = np.sqrt(this**2 + other**2)

            if isinstance(result, Quantity):
                # In case we worked with quantities we need to return the
                # uncertainty that has the same unit as the resulting data
                if result.unit == result_data.unit:
                    return result.value
                else:
                    # Convert it to the data's unit and then drop the unit.
                    return result.to(result_data.unit).value
            else:
                return result

    def _propagate_subtract(self, other_uncert, result_data, correlation):
        # Since the formulas are equivalent to addition you should look at the
        # explanations provided in _propagate_add

        if self.array is None:
            if other_uncert.effective_unit is not None and (
                        result_data.unit != other_uncert.effective_unit):
                return (other_uncert.array * other_uncert.effective_unit).to(
                            result_data.unit).value
            else:
                return deepcopy(other_uncert.array)
        elif other_uncert.array is None:
            if self.effective_unit is not None and self.effective_unit != self.parent_nddata.unit:
                return (self.array * self.effective_unit).to(result_data.unit).value
            else:
                return deepcopy(self.array)
        else:
            # Formula: sigma = sqrt(dA**2 + dB**2 - 2*cor*dA*dB)
            if self.effective_unit != other_uncert.effective_unit:
                this = self.array * self.effective_unit
                other = other_uncert.array * other_uncert.effective_unit
            else:
                this = self.array
                other = other_uncert.array
            if isinstance(correlation, np.ndarray) or correlation != 0:
                corr = 2 * correlation * this * other
                # The only difference to addition is that the correlation is
                # subtracted.
                result = np.sqrt(this**2 + other**2 - corr)
            else:
                result = np.sqrt(this**2 + other**2)
            if isinstance(result, Quantity):
                if result.unit == result_data.unit:
                    return result.value
                else:
                    return result.to(result_data.unit).value
            else:
                return result

    def _propagate_multiply(self, other_uncert, result_data, correlation):

        # For multiplication we don't need the result as quantity
        if isinstance(result_data, Quantity):
            result_data = result_data.value

        if self.array is None:
            # Formula: sigma = |A| * dB

            # We want the result to have the same unit as the result so we
            # only need to convert the unit of the other uncertainty if it is
            # different from it's datas unit.
            if other_uncert.effective_unit != other_uncert.parent_nddata.unit:
                other = (other_uncert.array * other_uncert.effective_unit).to(
                            other_uncert.parent_nddata.unit).value
            else:
                other = other_uncert.array
            return np.abs(self.parent_nddata.data * other)

        elif other_uncert.array is None:
            # Formula: sigma = dA * |B|

            # Just the reversed case
            if self.effective_unit != self.parent_nddata.unit:
                this = (self.array * self.effective_unit).to(
                                            self.parent_nddata.unit).value
            else:
                this = self.array
            return np.abs(other_uncert.parent_nddata.data * this)

        else:
            # Formula: sigma = |AB|*sqrt((dA/A)**2+(dB/B)**2+2*dA/A*dB/B*cor)

            # This formula is not very handy since it generates NaNs for every
            # zero in A and B. So we rewrite it:

            # Formula: sigma = sqrt((dA*B)**2 + (dB*A)**2 + (2 * cor * ABdAdB))

            # Calculate: dA * B (left)
            if self.effective_unit != self.parent_nddata.unit:
                # To get the unit right we need to convert the unit of
                # each uncertainty to the same unit as it's parent
                left = ((self.array * self.effective_unit).to(
                        self.parent_nddata.unit).value *
                        other_uncert.parent_nddata.data)
            else:
                left = self.array * other_uncert.parent_nddata.data

            # Calculate: dB * A (right)
            if other_uncert.effective_unit != other_uncert.parent_nddata.unit:
                right = ((other_uncert.array * other_uncert.effective_unit).to(
                        other_uncert.parent_nddata.unit).value *
                        self.parent_nddata.data)
            else:
                right = other_uncert.array * self.parent_nddata.data

            if isinstance(correlation, np.ndarray) or correlation != 0:
                corr = (2 * correlation * left * right)
                return np.sqrt(left**2 + right**2 + corr)
            else:
                return np.sqrt(left**2 + right**2)

    def _propagate_divide(self, other_uncert, result_data, correlation):

        # For division we don't need the result as quantity
        if isinstance(result_data, Quantity):
            result_data = result_data.value

        if self.array is None:
            # Formula: sigma = |(A / B) * (dB / B)|

            # Calculate: dB / B (right)
            if other_uncert.effective_unit != other_uncert.parent_nddata.unit:
                # We need (dB / B) to be dimensionless so we convert
                # (if necessary) dB to the same unit as B
                right = ((other_uncert.array * other_uncert.effective_unit).to(
                    other_uncert.parent_nddata.unit).value /
                    other_uncert.parent_nddata.data)
            else:
                right = (other_uncert.array / other_uncert.parent_nddata.data)
            return np.abs(result_data * right)

        elif other_uncert.array is None:
            # Formula: sigma = dA / |B|.

            # Calculate: dA
            if self.effective_unit != self.parent_nddata.unit:
                # We need to convert dA to the unit of A to have a result that
                # matches the resulting data's unit.
                left = (self.array * self.effective_unit).to(
                        self.parent_nddata.unit).value
            else:
                left = self.array

            return np.abs(left / other_uncert.parent_nddata.data)

        else:
            # Formula: sigma = |A/B|*sqrt((dA/A)**2+(dB/B)**2-2*dA/A*dB/B*cor)

            # As with multiplication this formula creates NaNs where A is zero.
            # So I'll rewrite it again:
            # => sigma = sqrt((dA/B)**2 + (AdB/B**2)**2 - 2*cor*AdAdB/B**3)

            # So we need to calculate dA/B in the same units as the result
            # and the dimensionless dB/B to get a resulting uncertainty with
            # the same unit as the data.

            # Calculate: dA/B (left)
            if self.effective_unit != self.parent_nddata.unit:
                left = ((self.array * self.effective_unit).to(
                        self.parent_nddata.unit).value /
                        other_uncert.parent_nddata.data)
            else:
                left = self.array / other_uncert.parent_nddata.data

            # Calculate: dB/B (right)
            if other_uncert.effective_unit != other_uncert.parent_nddata.unit:
                right = ((other_uncert.array * other_uncert.effective_unit).to(
                    other_uncert.parent_nddata.unit).value /
                    other_uncert.parent_nddata.data) * result_data
            else:
                right = (result_data * other_uncert.array /
                         other_uncert.parent_nddata.data)

            if isinstance(correlation, np.ndarray) or correlation != 0:
                corr = 2 * correlation * left * right
                # This differs from multiplication because the correlation
                # term needs to be subtracted
                return np.sqrt(left**2 + right**2 - corr)
            else:
                return np.sqrt(left**2 + right**2)
