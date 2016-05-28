# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


from astropy import log


__all__ = ['as_integer', 'as_unsigned_integer']


CONV_FAILED = '{0} cannot be converted to {1}.'
LOG_MSG = '{0} is not a {1} but was converted to {2}.'
EXCEPT_MSG = '{0} is not a {1}.'


def as_integer(value, info=True):
    """Checks if the value is an integer.

    Parameters
    ----------
    value : any type
        The value to investigate.

    info : `bool`, optional
        If ``True`` print an info message rather than throw an exception
        (``False``) if it is convertable to an integer but not identical.
        Default is ``True``.

    Returns
    -------
    value : `int`
        The value as integer.

    Raises
    ------
    ValueError
        If the ``value`` cannot be converted to an integer or ``log=False`` and
        the value wasn't an integer.

    Notes
    -----
    Generally it is frowned upon to do explicit type checks, the more
    appropriate way is to do try / excepts or just try to do it and let it
    fail. But given that some functions are really convolved and the standard
    exception messages would be unhelpful this function might become
    appropriate but definitly not always!

    Examples
    --------
    For example if some variable must be an integer but due to some
    calculations it might be a floating point value that is equal to an integer
    this function will not print any warning or exception::

        >>> from nddata.utils.inputvalidation import as_integer
        >>> as_integer(2.0)
        2

    but in case the value differs but is convertible to an integer a warning is
    printed (or an Exception raised if ``info=False`` was given)::

        >>> as_integer(3.1)
        INFO: 3.1 cannot be converted to integer. \
[nddata.utils.inputvalidation]
        3
    """
    typ = 'integer'
    # Try to cast it to int if that fails we can only reraise the Error message
    try:
        value_ = int(value)
    except (ValueError, TypeError):
        raise ValueError(CONV_FAILED.format(value, typ))

    # If the value differs (for example 1.1 -> 1) but not when the real value
    # stayed the same (for example 1.0 -> 1) we either raise an Info-message
    # that the value was interpreted as something else or raise an Exception.
    if value != value_:
        if info:
            log.info(CONV_FAILED.format(value, typ, value_))
        else:
            raise ValueError(EXCEPT_MSG.format(value, typ))

    return value_


def as_unsigned_integer(value, info=True):
    """Checks if the value is an unsigned integer.

    Parameters
    ----------
    value : any type
        The value to investigate.

    info : `bool`, optional
        If ``True`` print an info message rather than throw an exception
        (``False``) if it is convertable to an unsigned integer but not
        identical.
        Default is ``True``.

    Returns
    -------
    value : `int`
        The value as unsigned integer.

    Raises
    ------
    ValueError
        If the ``value`` cannot be converted to an integer or ``log=False`` and
        the value wasn't an unsigned integer.

    Notes
    -----
    Generally it is frowned upon to do explicit type checks, the more
    appropriate way is to do try / excepts or just try to do it and let it
    fail. But given that some functions are really convolved and the standard
    exception messages would be unhelpful this function might become
    appropriate but definitly not always!
    """
    typ = 'unsigned integer'
    # Try to cast it to int and then take the absolute of it. In case this
    # fails with a ValueError we reraise an Exception.
    try:
        value_ = abs(int(value))
    except (ValueError, TypeError):
        raise ValueError(CONV_FAILED.format(value, typ))

    # Same as for integer. Check if the value is nominally the same and print
    # an info or exception if not.
    if value != value_:
        if info:
            log.info(CONV_FAILED.format(value, typ, value_))
        else:
            raise ValueError(EXCEPT_MSG.format(value, typ))

    return value_
