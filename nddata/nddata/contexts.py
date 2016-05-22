# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from ..utils.contextmanager import DictionaryContext

__all__ = ['ContextArithmeticDefaults']


class ContextArithmeticDefaults(DictionaryContext):
    """This class contains the default arguments for \
            `~.mixins.NDArithmeticMixin` and can be altered within contexts.

    The values only represent the defaults, so only apply if no explicit
    parameter was provided. Chaning any value will be global but it can be
    used as context manager to change them only temporary.

    Examples
    --------
    You want to calculate some values but want no uncertainty-propagation
    without overriding this at each call::

        >>> from nddata.nddata import NDData, VarianceUncertainty
        >>> from nddata.nddata import ContextArithmeticDefaults
        >>> ndd = NDData([1,2,3,4], uncertainty=VarianceUncertainty([1,1,1,1]))
        >>> with ContextArithmeticDefaults(propagate_uncertainties=None):
        ...     ndd2 = ndd.add(4).add(5).add(6).add(7)
        >>> print(ndd2.uncertainty)
        None

    This is equivalent to::

        >>> ndd2 = ndd.add(4, propagate_uncertainties=None
        ...               ).add(5, propagate_uncertainties=None
        ...                     ).add(6, propagate_uncertainties=None
        ...                           ).add(7, propagate_uncertainties=None)
        >>> print(ndd2.uncertainty)
        None

    But you can also change defaults within the context::

        >>> with ContextArithmeticDefaults() as defaults:
        ...     ndd2 = ndd.add(4).add(5).add(6).add(7)
        ...     defaults['propagate_uncertainties'] = None
        ...     ndd3 = ndd.add(4).add(5).add(6).add(7)
        ...     defaults['propagate_uncertainties'] = True
        ...     ndd4 = ndd.add(4).add(5).add(6).add(7)
        >>> print(ndd2.uncertainty)
        VarianceUncertainty([1, 1, 1, 1])
        >>> print(ndd3.uncertainty)
        None
        >>> print(ndd4.uncertainty)
        VarianceUncertainty([1, 1, 1, 1])

    The context-manager will clean up any changes after you exit it so you
    don't need to reset them manually afterwards.

    .. warning::
        You can also globally change the defaults by using
        ``ContextArithmeticDefaults.dct['parameter_name'] = new_default`` but
        these will be set permanently, if not reset manually.

    With the `~nddata.nddata.mixins.NDArithmeticPyOpsMixin` this context
    manager becomes important if you want to give any parameters to your
    operations::

        >>> from nddata.nddata import NDDataBase
        >>> from nddata.nddata.mixins import NDArithmeticPyOpsMixin
        >>> class NDData2(NDArithmeticPyOpsMixin, NDDataBase): pass

        >>> ndd = NDData2([1,2,3], meta={'something_very_important': 100})
        >>> with ContextArithmeticDefaults(handle_meta='ff') as defaults:
        ...     res = (2 + ndd * 5) / 7
        >>> res
        NDData2([ 1.        ,  1.71428571,  2.42857143])

        >>> res.meta
        {'something_very_important': 100}
    """
    dct = {'propagate_uncertainties': True,
           'handle_mask': np.logical_or,
           'handle_meta': None,
           'handle_flags': None,
           'compare_wcs': 'first_found',
           'uncertainty_correlation': 0}
