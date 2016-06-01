# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from copy import copy

from astropy.wcs import WCS


__all__ = ['do_copy']


def do_copy(value):
    """This function copies a value like `copy.copy` but special cases some \
            classes that have a broken or unexpected `copy.copy` behaviour.

    This affects the classes `astropy.wcs.WCS` and `astropy.io.fits.Header`.

    Parameters
    ----------
    value : any type
        The value to be copied.

    Returns
    -------
    copy : any type
        The copied value.

    Notes
    -----
    If the ``value`` defines a ``copy`` method it will be called. Probably it
    was defined because it's the most efficient way to copy the class. But the
    main reason for this function are some special cases:

    In case of `astropy.wcs.WCS` it will call :meth:`astropy.wcs.WCS.deepcopy`
    so also the attributes in the underlying `~astropy.wcs.Wcsprm` instance
    are copied (https://github.com/astropy/astropy/issues/4989).

    For `astropy.io.fits.Header` it will call
    :meth:`astropy.io.fits.Header.copy` because with a normal :func:`copy.copy`
    the instance is not really copied
    (https://github.com/astropy/astropy/issues/4990).
    """
    # Check if it has a copy-method (both exceptions have this method so we
    # don't bother regular values too much).
    if hasattr(value, 'copy'):
        # WCS instances must be deepcopied because the shallow copy only copies
        # the main instance and not the underlying attributes which are in my
        # opionion the most crucial ones to be copied!!!
        if isinstance(value, WCS):
            return value.deepcopy()
        # The Header object actually does the right thing when trying to copy
        # it. So this elif is commented, because we will always return the
        # result of the copy-method instead of applying the copy function on
        # the value. It is left here in case I need to modify the behaviour for
        # classes that have a copy method but do silly stuff there (like WCS).
        # from astropy.io.fits import Header
        # elif isinstance(value, Header):
        #    return value.copy()
        return value.copy()

    # In case there is no copy method just use the copy-function from the copy
    # module. It should do the right thing (probably).
    return copy(value)
