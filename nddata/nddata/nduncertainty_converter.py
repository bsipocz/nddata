# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from astropy import log
import astropy.units as u

from .exceptions import IncompatibleUncertaintiesException
from .exceptions import MissingDataAssociationException

from .nduncertainty_unknown import UnknownUncertainty
from .nduncertainty_stddev import StdDevUncertainty
from .nduncertainty_relstd import RelativeUncertainty
from .nduncertainty_var import VarianceUncertainty


__all__ = ['UncertaintyConverter']


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

        +-----------------------+-----------------------+---------+----------+
        | Source                | Target                | Forward | Backward |
        +=======================+=======================+=========+==========+
        | `UnknownUncertainty`  | `StdDevUncertainty`   | Yes     | Yes      |
        +-----------------------+-----------------------+---------+----------+
        | `UnknownUncertainty`  | `VarianceUncertainty` | Yes     | Yes      |
        +-----------------------+-----------------------+---------+----------+
        | `UnknownUncertainty`  | `RelativeUncertainty` | Yes     | Yes      |
        +-----------------------+-----------------------+---------+----------+
        | `StdDevUncertainty`   | `VarianceUncertainty` | Yes     | Yes      |
        +-----------------------+-----------------------+---------+----------+
        | `StdDevUncertainty`   | `RelativeUncertainty` | Yes     | Yes      |
        +-----------------------+-----------------------+---------+----------+
        | `VarianceUncertainty` | `RelativeUncertainty` | Yes     | Yes      |
        +-----------------------+-----------------------+---------+----------+

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
to or from an UnknownUncertainty. [nddata.nddata.nduncertainty_converter]
            >>> unc2.data
            array([ 1.,  1.,  1.,  1.,  1.])
        """
        try:
            return cls._converter[(source, target)]
        except KeyError:
            msg = "cannot convert {0} to {1}".format(source.__name__,
                                                     target.__name__)
            raise IncompatibleUncertaintiesException(msg)


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
UncertaintyConverter.register(UnknownUncertainty, VarianceUncertainty,
                              _convert_unknown_to_something,
                              _convert_unknown_to_something)
UncertaintyConverter.register(UnknownUncertainty, RelativeUncertainty,
                              _convert_unknown_to_something,
                              _convert_unknown_to_something)


def _convert_std_to_var(val):
    data = val.data
    unit = val.unit
    try:
        parent_nddata = val.parent_nddata
    except MissingDataAssociationException:
        parent_nddata = None
    if data is not None:
        data = data ** 2
    if unit is not None:
        unit = unit ** 2
    return {'data': data, 'unit': unit, 'parent_nddata': parent_nddata}


def _convert_var_to_std(val):
    data = val.data
    unit = val.unit
    try:
        parent_nddata = val.parent_nddata
    except MissingDataAssociationException:
        parent_nddata = None
    if data is not None:
        data = data ** (1/2)
    if unit is not None:
        unit = unit ** (1/2)
    return {'data': data, 'unit': unit, 'parent_nddata': parent_nddata}


UncertaintyConverter.register(StdDevUncertainty, VarianceUncertainty,
                              _convert_std_to_var,
                              _convert_var_to_std)


def _convert_std_to_rel(val):
    try:
        parent_nddata = val.parent_nddata
    except MissingDataAssociationException:
        msg = 'converting to relative uncertainty requires the parents data.'
        raise MissingDataAssociationException(msg)

    # We need the parents values to calculate the relative ones.
    if parent_nddata.unit is not None:
        data_p = parent_nddata.data * parent_nddata.unit
    else:
        data_p = parent_nddata.data

    if val.effective_unit is not None:
        data_u = val.data * val.effective_unit
    else:
        data_u = val.data

    data = data_u / data_p

    if isinstance(data, u.Quantity):
        data = data.to(u.dimensionless_unscaled).value
    return {'data': data, 'unit': None, 'parent_nddata': parent_nddata}


def _convert_rel_to_std(val):
    try:
        parent_nddata = val.parent_nddata
    except MissingDataAssociationException:
        msg = 'converting from relative uncertainty requires the parents data.'
        raise MissingDataAssociationException(msg)

    # We need the parents values to calculate the relative ones.
    data_p = parent_nddata.data

    data_u = val.data

    data = data_p * data_u

    return {'data': data, 'unit': None, 'parent_nddata': parent_nddata}


UncertaintyConverter.register(StdDevUncertainty, RelativeUncertainty,
                              _convert_std_to_rel,
                              _convert_rel_to_std)


def _convert_var_to_rel(val):
    try:
        parent_nddata = val.parent_nddata
    except MissingDataAssociationException:
        msg = 'converting to relative uncertainty requires the parents data.'
        raise MissingDataAssociationException(msg)

    # We need the parents values to calculate the relative ones.
    if parent_nddata.unit is not None:
        data_p = (parent_nddata.data * parent_nddata.unit)**2
    else:
        data_p = parent_nddata.data ** 2

    if val.effective_unit is not None:
        data_u = val.data * val.effective_unit
    else:
        data_u = val.data

    data = np.sqrt(data_u / data_p)

    if isinstance(data, u.Quantity):
        data = data.to(u.dimensionless_unscaled).value
    return {'data': data, 'unit': None, 'parent_nddata': parent_nddata}


def _convert_rel_to_var(val):
    try:
        parent_nddata = val.parent_nddata
    except MissingDataAssociationException:
        msg = 'converting from relative uncertainty requires the parents data.'
        raise MissingDataAssociationException(msg)

    # We need the parents values to calculate the relative ones.
    data_p = parent_nddata.data

    data_u = val.data

    data = (data_p * data_u)**2

    return {'data': data, 'unit': None, 'parent_nddata': parent_nddata}


UncertaintyConverter.register(VarianceUncertainty, RelativeUncertainty,
                              _convert_var_to_rel,
                              _convert_rel_to_var)
