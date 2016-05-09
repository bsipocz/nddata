# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Sundry function and class decorators."""

from __future__ import print_function

from astropy.extern import six


__all__ = ['format_doc']


def format_doc(docstring, *args, **kwargs):
    """
    Replaces the docstring of the decorated object and then formats it.

    The formatting works like :meth:`str.format` and if the decorated object
    already has a docstring this docstring can be included in the new
    documentation if you use the ``{__doc__}`` placeholder.
    Its primary use is for reusing a *long* docstring in multiple functions
    when it is the same or only slightly different between them.

    Parameters
    ----------
    docstring : str or object or None
        The docstring that will replace the docstring of the decorated
        object. If it is an object like a function or class it will
        take the docstring of this object. If it is a string it will use the
        string itself. One special case is if the string is ``None`` then
        it will use the decorated functions docstring and formats it.

    args :
        passed to :meth:`str.format`.

    kwargs :
        passed to :meth:`str.format`. If the function has a (not empty)
        docstring the original docstring is added to the kwargs with the
        keyword ``'__doc__'``.

    Raises
    ------
    ValueError
        If the ``docstring`` (or interpreted docstring if it was ``None``
        or not a string) is empty.

    IndexError, KeyError
        If a placeholder in the (interpreted) ``docstring`` was not filled. see
        :meth:`str.format` for more information.

    Notes
    -----
    Using this decorator allows, for example Sphinx, to parse the
    correct docstring.

    Examples
    --------

    Replacing the current docstring is very easy::

        >>> from nddata.utils.decorators import format_doc
        >>> @format_doc('''Perform num1 + num2''')
        ... def add(num1, num2):
        ...     return num1+num2
        ...
        >>> help(add) # doctest: +SKIP
        Help on function add in module __main__:
        <BLANKLINE>
        add(num1, num2)
            Perform num1 + num2

    sometimes instead of replacing you only want to add to it::

        >>> doc = '''
        ...       {__doc__}
        ...       Parameters
        ...       ----------
        ...       num1, num2 : Numbers
        ...       Returns
        ...       -------
        ...       result: Number
        ...       '''
        >>> @format_doc(doc)
        ... def add(num1, num2):
        ...     '''Perform addition.'''
        ...     return num1+num2
        ...
        >>> help(add) # doctest: +SKIP
        Help on function add in module __main__:
        <BLANKLINE>
        add(num1, num2)
            Perform addition.
            Parameters
            ----------
            num1, num2 : Numbers
            Returns
            -------
            result: Number

    in case one might want to format it further::

        >>> doc = '''
        ...       Perform {0}.
        ...       Parameters
        ...       ----------
        ...       num1, num2 : Numbers
        ...       Returns
        ...       -------
        ...       result: Number
        ...           result of num1 {op} num2
        ...       {__doc__}
        ...       '''
        >>> @format_doc(doc, 'addition', op='+')
        ... def add(num1, num2):
        ...     return num1+num2
        ...
        >>> @format_doc(doc, 'subtraction', op='-')
        ... def subtract(num1, num2):
        ...     '''Notes: This one has additional notes.'''
        ...     return num1-num2
        ...
        >>> help(add) # doctest: +SKIP
        Help on function add in module __main__:
        <BLANKLINE>
        add(num1, num2)
            Perform addition.
            Parameters
            ----------
            num1, num2 : Numbers
            Returns
            -------
            result: Number
                result of num1 + num2
        >>> help(subtract) # doctest: +SKIP
        Help on function subtract in module __main__:
        <BLANKLINE>
        subtract(num1, num2)
            Perform subtraction.
            Parameters
            ----------
            num1, num2 : Numbers
            Returns
            -------
            result: Number
                result of num1 - num2
            Notes: This one has additional notes.

    These methods can be combined an even taking the docstring from another
    object is possible as docstring attribute. You just have to specify the
    object::

        >>> @format_doc(add)
        ... def another_add(num1, num2):
        ...     return num1 + num2
        ...
        >>> help(another_add) # doctest: +SKIP
        Help on function another_add in module __main__:
        <BLANKLINE>
        another_add(num1, num2)
            Perform addition.
            Parameters
            ----------
            num1, num2 : Numbers
            Returns
            -------
            result: Number
                result of num1 + num2

    But be aware that this decorator *only* formats the given docstring not
    the strings passed as ``args`` or ``kwargs`` (not even the original
    docstring)::

        >>> @format_doc(doc, 'addition', op='+')
        ... def yet_another_add(num1, num2):
        ...    '''This one is good for {0}.'''
        ...    return num1 + num2
        ...
        >>> help(yet_another_add) # doctest: +SKIP
        Help on function yet_another_add in module __main__:
        <BLANKLINE>
        yet_another_add(num1, num2)
            Perform addition.
            Parameters
            ----------
            num1, num2 : Numbers
            Returns
            -------
            result: Number
                result of num1 + num2
            This one is good for {0}.

    To work around it you could specify the docstring to be ``None``::

        >>> @format_doc(None, 'addition')
        ... def last_add_i_swear(num1, num2):
        ...    '''This one is good for {0}.'''
        ...    return num1 + num2
        ...
        >>> help(last_add_i_swear) # doctest: +SKIP
        Help on function last_add_i_swear in module __main__:
        <BLANKLINE>
        last_add_i_swear(num1, num2)
            This one is good for addition.

    Using it with ``None`` as docstring allows to use the decorator twice
    on an object to first parse the new docstring and then to parse the
    original docstring or the ``args`` and ``kwargs``.
    """
    def set_docstring(obj):
        if docstring is None:
            # None means: use the objects __doc__
            doc = obj.__doc__
            # Delete documentation in this case so we don't end up with
            # awkwardly self-inserted docs.
            obj.__doc__ = None
        elif isinstance(docstring, six.string_types):
            # String: use the string that was given
            doc = docstring
        else:
            # Something else: Use the __doc__ of this
            doc = docstring.__doc__

        if not doc:
            # In case the docstring is empty it's probably not what was wanted.
            raise ValueError('docstring must be a string or containing a '
                             'docstring that is not empty.')

        # If the original has a not-empty docstring append it to the format
        # kwargs.
        kwargs['__doc__'] = obj.__doc__ or ''
        if six.PY2 and isinstance(obj, type):
            # For python 2 we must create a subclass because there the __doc__
            # is not mutable for objects.
            obj = type(obj.__name__, (obj,), {'__doc__': doc.format(*args, **kwargs),
                                              '__module__': obj.__module__})
        else:
            obj.__doc__ = doc.format(*args, **kwargs)
        return obj
    return set_docstring
