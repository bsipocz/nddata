# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from .meta import NDUncertainty

__all__ = ['UnknownUncertainty']


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

    def convert_unit_to(self, unit, equivalencies=[]):
        """Returns an uncertainty converted to the new unit.

        This method requires that the `unit` is not ``None``.

        See also :meth:`~nddata.nddata.mixins.NDUnitConvMixin.convert_unit_to`.

        Raises
        ------
        AttributeError
            If the uncertainty has no unit.
        """
        # If the unit is None raise an AttributeError. This makes it easy for
        # the "convert_unit_to" from the parent to catch this error.
        if self.unit is None:
            raise AttributeError('{0} has no unit and cannot be '
                                 'converted.'.format(self.__class__.__name__))
        conv_data = self.unit.to(unit, self.data, equivalencies)
        return self.__class__(conv_data, unit, copy=False)
