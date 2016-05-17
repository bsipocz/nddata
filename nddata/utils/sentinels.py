# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


from astropy.extern import six


__all__ = ['SentinelFactory', 'ParameterNotSpecified']


class SentinelFactory(object):
    """Creates instances that act like `None`.

    Parameters
    ----------
    name : `str`
        The representation for the sentinel. If it is printed somewhere.

    Examples
    --------
    To create a sentinel::

        >>> from nddata.utils.sentinels import SentinelFactory
        >>> anotherNone = SentinelFactory('None2')
        >>> anotherNone
        None2

    The primary purpose is that it will always be the same instance, so for
    example if copied it will remain the same::

        >>> from copy import copy
        >>> yetanotherNone = copy(anotherNone)
        >>> yetanotherNone is anotherNone
        True

    It will like `None` evaluate to `False` in a boolean context::

        >>> if anotherNone:
        ...     print('this will never happen.')

    There are some builtin sentinels already implemented:

    - ``ParameterNotSpecified``: is a sentinal useable for arguments where
      `None` would be an allowed value and you need a value that tells you it
      wasn't given::

          >>> from nddata.utils.sentinels import ParameterNotSpecified
          >>> def func(value=ParameterNotSpecified):
          ...     if value is ParameterNotSpecified:
          ...         print("The parameter was not given.")
          ...     else:
          ...         return str(value)

          >>> func(10)
          '10'
          >>> func(None)
          'None'
          >>> func()
          The parameter was not given.
          >>> help(func)  # function signature uses "unspecified" as default
          Help on function func in module nddata.utils.sentinels:
          <BLANKLINE>
          func(value=unspecified)
          <BLANKLINE>

    """
    __slots__ = ('_name', )

    def __init__(self, name):
        if not isinstance(name, six.string_types):
            raise TypeError('name must be a string and not '
                            'a "{0}"'.format(name.__class__.__name__))
        object.__setattr__(self, '_name', name)

    # Representation and casting to strings
    def __repr__(self):
        return self._name

    def __str__(self):
        return self._name

    # The sentinal should evaluate to False
    # Python3
    def __bool__(self):
        return False

    # Python2
    def __nonzero__(self):
        return False

    # No copy of instances, just return self
    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    # TODO: Probably with pickling it could be modified. No need to do that
    # just now...

    # The name attribute should be unchangable
    def __setattr__(self, *args, **kwargs):
        raise TypeError('{0} cannot be modified.'
                        ''.format(self.__class__.__name__))

    def __delattr__(self, *args, **kwargs):
        raise TypeError('{0} cannot be modified.'
                        ''.format(self.__class__.__name__))


ParameterNotSpecified = SentinelFactory('unspecified')
