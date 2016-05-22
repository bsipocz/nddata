# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from copy import deepcopy
from ..meta.nduncertainty_meta import NDUncertainty


__all__ = ['NDUnitConvMixin']


class NDUnitConvMixin(object):
    """Mixin class to add unit conversions to an `~.NDDataBase` object.

    The :meth:`convert_unit_to` will convert the appropriate attributes and
    return a new class containing them and the unaffected attributes as copy.
    """

    def convert_unit_to(self, unit, equivalencies=[]):
        """Returns a new `NDDataBase` object whose values have been converted
        to a new unit.

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
            The converted dataset. The from the conversion unaffected
            attributes like ``meta`` and ``mask`` are copied.

        Raises
        ------
        UnitConversionError
            If units are inconsistent.
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
        return deepcopy(self.mask)

    def _convert_unit_meta(self, unit, equivalencies):
        return deepcopy(self.meta)

    def _convert_unit_wcs(self, unit, equivalencies):
        if self.wcs is None:
            return None
        return deepcopy(self.wcs)

    def _convert_unit_flags(self, unit, equivalencies):
        return deepcopy(self.flags)

    def _convert_unit_uncertainty(self, unit, equivalencies):
        if self.uncertainty is None:
            return None
        elif hasattr(self.uncertainty, 'convert_unit_to'):
            return self.uncertainty.convert_unit_to(unit=unit,
                                                    equivalencies=equivalencies)
        else:
            # just return it, they don't support this API. :-)
            return deepcopy(self.uncertainty)
