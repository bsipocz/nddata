.. _nddata_unit_conversion:

Converting the unit of NDData
=============================

Introduction
------------

Converting the unit of an `~nddata.nddata.NDDataBase` only differs slightly
from :meth:`astropy.units.Quantity.to` because it also converts the unit of
the ``uncertainty``.

The appropriate Mixin to enable unit conversion is
`~nddata.nddata.mixins.NDUnitConvMixin` but this is already included in
`~nddata.nddata.NDData`::

    >>> from nddata.nddata import NDData
    >>> ndd = NDData([100, 200], unit='cm')
    >>> ndd.convert_unit_to('m')
    NDData([ 1.,  2.])

It also converts the uncertainy based on it's unit::

    >>> from nddata.nddata import VarianceUncertainty
    >>> ndd = NDData([100, 200], unit='cm',
    ...              uncertainty=VarianceUncertainty(5))
    >>> ndd2 = ndd.convert_unit_to('m')
    >>> ndd2.uncertainty
    VarianceUncertainty(0.0005)

Depending on the uncertainty type different rules apply:

- `~nddata.nddata.UnknownUncertainty` can only be converted if it was
  explicitly created with a unit.
- `~nddata.nddata.RelativeUncertainty` cannot be converted but doesn't raise
  an Error. It will just stay the same.
- `~nddata.nddata.StdDevUncertainty` will be converted based on it's own unit
  or failing that assumes that it has the same unit as it's parent.
- `~nddata.nddata.VarianceUncertainty` will be converted based on it's own unit
  or failing that assumes that it has the squared unit of it's parent.
  For variance it will also convert it to the squared unit of the input.

The example above highlights the special behaviour: The uncertainty had no
explicit unit so it assumed it had the squared unit of it's parent (``m**2``)
and converted the unit to the squared input (``cm**2``) thus the uncertainty
was divided by 10000.
