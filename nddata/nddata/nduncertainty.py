# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from astropy import log
from astropy.units import Quantity
import astropy.units as u

from .meta import NDUncertainty, NDUncertaintyGaussian
from .exceptions import IncompatibleUncertaintiesException
from .exceptions import MissingDataAssociationException

__all__ = ['StdDevUncertainty', 'UnknownUncertainty', 'UncertaintyConverter']


class UncertaintyConverter(object):
    """Registry class to manage possible conversions between \
            `~.meta.NDUncertainty`-like classes.

    Only registered direct conversions are possible, this class will make no
    attempt at doing implicit intermediate conversions.
    """
    _converter = {}

    @classmethod
    def register(cls, source, target, forward, backward):
        """Register another conversion.

        Parameters
        ----------
        source, target : classes
            The source and the target class. Must be classes not instances.

        forward, backward : callables
            Functions that convert an instance to a `dict` that can be used to
            create an instance of the other class. Forward is the conversion
            from source to target and backward the conversion from target to
            source.

        Examples
        --------
        A simplified example that converts the data of an `UnknownUncertainty`
        to `StdDevUncertainty` without any conversions::

            >>> def unknown_to_from(uncertainty):
            ...     return {'data': uncertainty.data}

        The register it I choose another subclass of UnknownUncertainty so
        it doesn't mess up the (better) pre-defined conversion::

            >>> from nddata.nddata import UncertaintyConverter
            >>> from nddata.nddata import UnknownUncertainty, StdDevUncertainty
            >>> class UnknownUncert(UnknownUncertainty): pass
            >>> UncertaintyConverter.register(UnknownUncert,
            ...     StdDevUncertainty, unknown_to_from, unknown_to_from)

        and afterwards you can convert from and to `UnknownUncertainty`::

            >>> import numpy as np
            >>> uncertainty = UnknownUncert(np.array([10]))
            >>> StdDevUncertainty(uncertainty)
            StdDevUncertainty([10])

            >>> StdDevUncertainty.from_uncertainty(uncertainty)
            StdDevUncertainty([10])

        The other way around works too::

            >>> uncertainty = StdDevUncertainty(np.array([10]))
            >>> UnknownUncert(uncertainty)
            UnknownUncert([10])

        .. warning::
            You can overwrite existing conversions if you need to customize the
            behaviour. But be careful you do not overwrite by accident.
        """
        cls._converter[(source, target)] = forward
        cls._converter[(target, source)] = backward

    @classmethod
    def get_converter_func(cls, source, target):
        """Returns the appropriate conversion function for the specified \
                source and target.

        .. note::
            This method is called by the function
            :meth:`~.NDUncertainty.from_uncertainty` and during initialization
            of a `~.NDUncertainty`-like class. So normally you don't need to
            use this method directly.

        Parameters
        ----------
        source, target : classes
            The source and the target class. Must be classes not instances.

        Returns
        -------
        conversion_func : callable
            A callable that returns a `dict` that can be used to construct a
            new instance of the target class.

        Notes
        -----
        Possible conversions:

        +----------------------+-----------------------+---------+----------+
        | Source               | Target                | Forward | Backward |
        +======================+=======================+=========+==========+
        | `UnknownUncertainty` | `StdDevUncertainty`   | Yes     | Yes      |
        +----------------------+-----------------------+---------+----------+

        Examples
        --------
        The conversion from or to `UnknownUncertainty` will print a warning
        since it assumes that the conversion should keep the values and only
        wrap it in another class::

            >>> from nddata.nddata import StdDevUncertainty, UnknownUncertainty
            >>> import numpy as np
            >>> unc1 = UnknownUncertainty(np.ones(5), unit='m')
            >>> unc2 = StdDevUncertainty.from_uncertainty(unc1)
            INFO: Assume the uncertainty values stay the same when converting \
to or from an UnknownUncertainty. [nddata.nddata.nduncertainty]
            >>> unc2.data
            array([ 1.,  1.,  1.,  1.,  1.])
        """
        try:
            return cls._converter[(source, target)]
        except KeyError:
            msg = "cannot convert {0} to {1}".format(source.__name__,
                                                     target.__name__)
            raise IncompatibleUncertaintiesException(msg)


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
    """Standard deviation uncertainty assuming first order gaussian error \
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

    the uncertainty ``data`` can also be set directly::

        >>> ndd.uncertainty.data = 2
        >>> ndd.uncertainty
        StdDevUncertainty(2)
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

        return self.__class__(result, copy=False)

    def _propagate_add(self, other_uncert, result_data, correlation):

        if self.data is None:
            # Formula: sigma = dB

            if other_uncert.effective_unit is not None and (
                        result_data.unit != other_uncert.effective_unit):
                # If the other uncertainty has a unit and this unit differs
                # from the unit of the result convert it to the results unit
                return other_uncert.unit.to(result_data.unit,
                                            other_uncert.data)
            else:
                # Copy the result because _propagate will not copy it but for
                # arithmetic operations users will expect copys.
                return other_uncert.data.copy()

        elif other_uncert.data is None:
            # Formula: sigma = dA

            if (self.effective_unit is not None and
                    self.effective_unit != self.parent_nddata.unit):
                # If the uncertainty has a different unit than the result we
                # need to convert it to the results unit.
                return self.unit.to(result_data.unit, self.data)
            else:
                # Copy the result because _propagate will not copy it but for
                # arithmetic operations users will expect copys.
                return self.data.copy()

        else:
            # Formula: sigma = sqrt(dA**2 + dB**2 + 2*cor*dA*dB)

            # Calculate: dA (this) and dB (other)
            if self.effective_unit != other_uncert.effective_unit:
                # In case the two uncertainties (or data) have different units
                # we need to use quantity operations. The case where only one
                # has a unit and the other doesn't is not possible with
                # addition and would have raised an exception in the data
                # computation
                this = self.data * self.effective_unit
                other = other_uncert.data * other_uncert.effective_unit
            else:
                # Since both units are the same or None we can just use
                # numpy operations
                this = self.data
                other = other_uncert.data

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

        if self.data is None:
            if other_uncert.effective_unit is not None and (
                        result_data.unit != other_uncert.effective_unit):
                return other_uncert.unit.to(result_data.unit,
                                            other_uncert.data)
            else:
                return other_uncert.data.copy()
        elif other_uncert.data is None:
            if (self.effective_unit is not None and
                    self.effective_unit != self.parent_nddata.unit):
                return self.unit.to(result_data.unit, self.data)
            else:
                return self.data.copy()
        else:
            # Formula: sigma = sqrt(dA**2 + dB**2 - 2*cor*dA*dB)
            if self.effective_unit != other_uncert.effective_unit:
                this = self.data * self.effective_unit
                other = other_uncert.data * other_uncert.effective_unit
            else:
                this = self.data
                other = other_uncert.data
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

        if self.data is None:
            # Formula: sigma = |A| * dB

            # We want the resulting uncertainty to have the same unit as the
            # result so we only need to convert the unit of the other
            # uncertainty if it is different from it's datas unit.
            if other_uncert.effective_unit != other_uncert.parent_nddata.unit:
                other = other_uncert.unit.to(other_uncert.parent_nddata.unit,
                                             other_uncert.data)
            else:
                other = other_uncert.data
            return np.abs(self.parent_nddata.data * other)

        elif other_uncert.data is None:
            # Formula: sigma = dA * |B|

            # Just the reversed case
            if self.effective_unit != self.parent_nddata.unit:
                this = self.unit.to(self.parent_nddata.unit, self.data)
            else:
                this = self.data
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
                first = self.unit.to(self.parent_nddata.unit, self.data)
            else:
                first = self.data

            first = first * other_uncert.parent_nddata.data

            # Calculate: dB * A (right)
            if other_uncert.effective_unit != other_uncert.parent_nddata.unit:
                second = other_uncert.unit.to(other_uncert.parent_nddata.unit,
                                              other_uncert.data)
            else:
                second = other_uncert.data
            second = second * self.parent_nddata.data

            if isinstance(correlation, np.ndarray) or correlation != 0:
                corr = (2 * correlation * first * second)
                return np.sqrt(first**2 + second**2 + corr)
            else:
                return np.sqrt(first**2 + second**2)

    def _propagate_divide(self, other_uncert, result_data, correlation):

        # For division we don't need the result as quantity
        if isinstance(result_data, Quantity):
            result_data = result_data.value

        if self.data is None:
            # Formula: sigma = |(A / B) * (dB / B)|

            # Calculate: dB / B (right)
            if other_uncert.effective_unit != other_uncert.parent_nddata.unit:
                # We need (dB / B) to be dimensionless so we convert
                # (if necessary) dB to the same unit as B
                dB = other_uncert.unit.to(other_uncert.parent_nddata.unit,
                                          other_uncert.data)
            else:
                dB = other_uncert.data
            return np.abs(result_data * dB / other_uncert.parent_nddata.data)

        elif other_uncert.data is None:
            # Formula: sigma = dA / |B|.

            # Calculate: dA
            if self.effective_unit != self.parent_nddata.unit:
                # We need to convert dA to the unit of A to have a result that
                # matches the resulting data's unit.
                dA = self.unit.to(self.parent_nddata.unit, self.data)
            else:
                dA = self.data

            return np.abs(dA / other_uncert.parent_nddata.data)

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
                first = self.unit.to(self.parent_nddata.unit, self.data)
            else:
                first = self.data
            first = first / other_uncert.parent_nddata.data

            # Calculate: dB/B (right)
            if other_uncert.effective_unit != other_uncert.parent_nddata.unit:
                second = other_uncert.unit.to(other_uncert.parent_nddata.unit,
                                              other_uncert.data)
            else:
                second = other_uncert.data
            second = result_data * second / other_uncert.parent_nddata.data

            if isinstance(correlation, np.ndarray) or correlation != 0:
                corr = 2 * correlation * first * second
                # This differs from multiplication because the correlation
                # term needs to be subtracted
                return np.sqrt(first**2 + second**2 - corr)
            else:
                return np.sqrt(first**2 + second**2)

    def _propagate_power(self, other_uncert, result_data, correlation):
        # Power is a bit tricky with units. But it boils down to:
        # - Exponent must be dimensionless or convertible to dimensionless
        # - Base can only hava a unit if the exponent is a scalar.
        # - if the exponent has an uncertainty the base must be dimensionless

        # Propagation for power can ignore the result's unit.
        if isinstance(result_data, Quantity):
            result_data = result_data.value

        if self.data is None:
            # only the operand has an uncertainty.

            # Formula: sigma = |A**B * ln(A) * dB|

            # We need to take the natural logarithm of A. This is only possible
            # if the unit is dimensionless.
            # The value here is named "A_" because even though it is A it is
            # only used to calculate the logarithm. The "A" for propagation
            # must not be converted otherwise we would lose the connection
            # to the result unit!
            if self.parent_nddata.unit is not None:
                # The base has some unit, try to convert it to dimensionless
                # which will bubble up an exception if not possible.
                A_ = self.parent_nddata.unit.to(u.dimensionless_unscaled,
                                                self.parent_nddata.data)
            else:
                # The base has no unit in that case just take the data.
                A_ = self.parent_nddata.data

            # FIXME: Absolute is not really in the formula. But the logarithm
            # fails for negative input so I thought it may be a good idea. :-)
            # See also the third case - there this formula is used too.
            lnA = np.log(np.abs(A_))

            # dB must be dimensionless because B is dimensionless so
            # convert it to dimensionless if any unit is present. This will
            # raise an exception if not possible. Or just take the data if
            # it has no unit.
            if other_uncert.effective_unit is not None:
                dB = other_uncert.effective_unit.to(u.dimensionless_unscaled,
                                                    other_uncert.data)
            else:
                dB = other_uncert.data

            return np.abs(result_data * lnA * dB)

        elif other_uncert.data is None:
            # Formula: sigma = | B * A ** (B-1) * dA |

            # To get the dimensions right we need to convert B to dimensionless
            # if it has any unit.
            if other_uncert.parent_nddata.unit is not None:
                B = other_uncert.parent_nddata.unit.to(
                        u.dimensionless_unscaled,
                        other_uncert.parent_nddata.data)
            else:
                B = other_uncert.parent_nddata.data

            # and dA must have the same unit as A so we don't need to struggle
            # with the dimensions later.
            if self.effective_unit != self.parent_nddata.unit:
                if self.parent_nddata.unit is None:
                    # The parent has no unit but the uncertainty has. Convert
                    # it to dimensionless.
                    dA = self.unit.to(u.dimensionless_unscaled, self.data)
                else:
                    # Both have units but different ones, convert the
                    # uncertainty to the unit of the parent
                    dA = self.unit.to(self.parent_nddata.unit, self.data)
            else:
                # Either the units are the same or both are dimensionless, just
                # take the uncertainty data.
                dA = self.data

            # A doesn't need it's unit because the formula ensures that the
            # result will have the same unit as the result.
            A = self.parent_nddata.data

            return np.abs(B * dA * A ** (B - 1))

        else:
            # Formula:
            # sigma = |A**B|*sqrt((BdA/A)**2+(ln(A)dB)**2+2Bln(A)dAdB*rho/A)

            # to allow for results where an element of A is zero we can
            # also write:
            # sigma = sqrt((BdAA**(B-1))**2 + (ln(A)dBA**B)**2 +
            #              2ln(A)BdAdBA**(2B-1)*rho)

            # to ensure we have the right dimension we convert some units, the
            # steps and explanations are the same as for the two special cases.

            if self.parent_nddata.unit is not None:
                A_ = self.parent_nddata.unit.to(u.dimensionless_unscaled,
                                                self.parent_nddata.data)
            else:
                A_ = self.parent_nddata.data

            lnA = np.log(np.abs(A_))

            if other_uncert.parent_nddata.unit is not None:
                B = other_uncert.parent_nddata.unit.to(
                        u.dimensionless_unscaled,
                        other_uncert.parent_nddata.data)
            else:
                B = other_uncert.parent_nddata.data

            if self.effective_unit != self.parent_nddata.unit:
                if self.parent_nddata.unit is None:
                    dA = self.unit.to(u.dimensionless_unscaled, self.data)
                else:
                    dA = self.unit.to(self.parent_nddata.unit, self.data)
            else:
                dA = self.data

            if other_uncert.effective_unit is not None:
                dB = other_uncert.effective_unit.to(u.dimensionless_unscaled,
                                                    other_uncert.data)
            else:
                dB = other_uncert.data

            A = self.parent_nddata.data

            # These two expressions are also present in the correlation term
            # so we can be a bit more efficient by calculating them only once.
            lnAdB = lnA * dB
            BdA = B * dA

            # The first two terms of the formula:
            first = BdA * A ** (B - 1)
            second = result_data * lnAdB

            # Check if any correlation is given and compute the appropriate
            # result.
            if isinstance(correlation, np.ndarray) or correlation != 0:
                third = 2 * correlation * lnAdB * BdA * A ** (2*B - 1)
                return np.sqrt(first**2 + second**2 + third)
            else:
                return np.sqrt(first**2 + second**2)


# Add conversions from different uncertainties
def _convert_unknown_to_something(val):
    log.info('Assume the uncertainty values stay the same when converting '
             'to or from an UnknownUncertainty.')
    data = val.data
    unit = val.unit
    try:
        parent_nddata = val.parent_nddata
    except MissingDataAssociationException:
        parent_nddata = None
    return {'data': data, 'unit': unit, 'parent_nddata': parent_nddata}

UncertaintyConverter.register(UnknownUncertainty, StdDevUncertainty,
                              _convert_unknown_to_something,
                              _convert_unknown_to_something)
