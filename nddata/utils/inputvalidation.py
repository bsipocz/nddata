# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from astropy import log


__all__ = ['as_integer', 'as_unsigned_integer', 'as_iterable', 'clamp']

inf = float('inf')


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


def clamp(value, minimum=-inf, maximum=inf):
    """Clamps the value to the range specified by minimum and maximum.

    Parameters
    ----------
    value : number
        The value to be clamped.

    minimum : number, optional
        The minimal value for the range.
        Default is ``-inf``.

    maximum : number
        The maximal value for the range.
        Default is ``inf``.

    Returns
    -------
    clamped_value : number
        The original value clamped to the range.

    See also
    --------
    numpy.clip : Clamp `numpy.ndarray`-like values.

    Examples
    --------
    Explicit ranges for clamping::

        >>> from nddata.utils.inputvalidation import clamp

        >>> clamp(1, 4, 10)
        4
        >>> clamp(6, 4, 10)
        6
        >>> clamp(14, 4, 10)
        10

    Clamp values to positive values::

        >>> clamp(10, 0)
        10
        >>> clamp(-10, 0)
        0

    Or equivalently to negative values::

        >>> clamp(10, maximum=0)
        0
        >>> clamp(-10, maximum=0)
        -10

    If you use them regularly remember that with :func:`functools.partial` you
    can create new functions by fixing one parameter::

        >>> from functools import partial
        >>> clamp_pos = partial(clamp, minimum=0)
        >>> clamp_neg = partial(clamp, maximum=0)
        >>> clamp_pos(-2)
        0
        >>> clamp_neg(2)
        0
    """
    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value

    # Other approaches (not tested for efficiency yet. Just what happened to
    # pass my mind and from:
    # http://stackoverflow.com/questions/4092528/how-to-clamp-an-integer-to-some-range-in-python
    # But the timings mentioned there might be out of date by now.

    # return sorted((minimum, value, maximum))[1]

    # return min(maximum, max(minimum, value))

    # return np.clip(value, minimum, maximum)


def as_iterable(value):
    """Checks if the value is an `collections.Iterable`.

    Parameters
    ----------
    value : any type
        The value to investigate.

    Returns
    -------
    value : `collections.Iterable` or `tuple`
        The value as iterable and if it wasn't an iterable it is now wrapped
        in a tuple.

    Notes
    -----
    Generally it is frowned upon to do explicit type checks. The more
    appropriate way is to do try / excepts or just try to do it and let it
    fail. But given that some functions are really convolved and the standard
    exception messages would be unhelpful this function might become
    appropriate but definitly not always!
    """
    # This function differs from the other as_* functions because we don't want
    # it to fail. Just convert it to a tuple if it's not an iterable.

    # Currently this is implemented to check if it has a length and if it
    # doesn't it's wrapped in a tuple.
    try:
        len(value)
    except:
        return (value, )
    else:
        return value

    # Other possibilities here:

    # if np.isscalar(value):
    #     return (value, )
    # else:
    #     return value

    # if hasattr(value, '__len__'):
    #     return value
    # else:
    #     return (value, )

    # if isinstance(value, collections.Iterable):
    #     return value
    # else:
    #     return (value, )

    # Advantages and Disadvantages:
    # np.isscalar clearly has problems if it comes to non-numericals. These
    # cases will probably be very rare so that might actually not be a problem.
    # isinstance is very slow for collections.abc because they check if a lot
    # of attributes are present. But subsequent calls with the same type should
    # be a lot faster because it registers types.
    # hasattr is clearly not the most pythonic way. But it should be fairly
    # fast and probably identical to the "try - except" option.

    # This function shouldn't be a bottleneck so it shouldn't matter which
    # approach is taken but IF it is then maybe checkout the other solutions.
