# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from copy import copy


__all__ = ['NDUnitConvMixin']


class NDUnitConvMixin(object):
    """Mixin class to add unit conversions to an `~nddata.nddata.NDDataBase` \
            object.

    The :meth:`convert_unit_to` will convert the appropriate attributes and
    return a new class containing them and the unaffected attributes as copy.
    """

    def convert_unit_to(self, unit, equivalencies=[]):
        """Returns a new `~nddata.nddata.NDDataBase` object whose values have \
            been converted to a new unit.

        Parameters
        ----------
        unit : `~astropy.units.Unit` or `str`
            The unit to convert to.

        equivalencies : list of equivalence pairs, optional
           A list of equivalence pairs in case the unit is not directly
           convertible. See :ref:`unit_equivalencies`.

        Returns
        -------
        result : `~nddata.nddata.NDDataBase`
            The converted dataset. The attributes of the result are copies of
            the initial attributes, even for unchanged attributes like
            ``meta``.

        Raises
        ------
        UnitConversionError
            If units are inconsistent.

        TypeError
            If the NDData has no unit.

        Examples
        --------
        This Mixin is already implemented in `~nddata.nddata.NDData`, so to
        use it just create an instance with a unit::

            >>> from nddata.nddata import NDData
            >>> ndd = NDData(100, unit='cm')
            >>> ndd.convert_unit_to('m')
            NDData(1.0)

        For further information about equivalencies checkout the `astropy
        documentation
        <http://docs.astropy.org/en/stable/units/equivalencies.html>`_.
        """
        # First check if the instance has a unit. It would fail if we tried to
        # convert None to anything.
        if self.unit is None:
            obj = self.__class__.__name__
            raise TypeError('cannot convert the unit of {0} because it has no '
                            'unit. To set the unit manually use "object.unit '
                            '= \'{1}\'"'.format(obj, unit))

        # Convert the attributes and catch the returned dictionary
        kwargs = self._convert_unit(unit, equivalencies)
        return self.__class__(copy=False, **kwargs)

    def _convert_unit(self, unit, equivalencies):
        # Create a dictionary holding the new attributes
        kwargs = {'unit': unit}
        # Use the Quantity framework to compute the new data values
        kwargs['data'] = self.unit.to(unit, self.data, equivalencies)
        kwargs['mask'] = self._convert_unit_mask(unit, equivalencies)
        kwargs['wcs'] = self._convert_unit_wcs(unit, equivalencies)
        kwargs['meta'] = self._convert_unit_meta(unit, equivalencies)
        kwargs['flags'] = self._convert_unit_flags(unit, equivalencies)
        kwargs['uncertainty'] = self._convert_unit_uncertainty(unit,
                                                               equivalencies)
        return kwargs

    def _convert_unit_mask(self, unit, equivalencies):
        if self.mask is None:
            return None
        return copy(self.mask)

    def _convert_unit_meta(self, unit, equivalencies):
        return copy(self.meta)

    def _convert_unit_wcs(self, unit, equivalencies):
        if self.wcs is None:
            return None
        return copy(self.wcs)

    def _convert_unit_flags(self, unit, equivalencies):
        return copy(self.flags)

    def _convert_unit_uncertainty(self, unit, equivalencies):
        if self.uncertainty is None:
            return None
        try:
            return self.uncertainty.convert_unit_to(unit=unit,
                                                    equivalencies=equivalencies)
        except AttributeError:
            # Either it has no "convert_unit_to" method OR it is an Unknown-
            # uncertainty without unit.
            return copy(self.uncertainty)
