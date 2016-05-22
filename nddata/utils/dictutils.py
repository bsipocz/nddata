# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from collections import defaultdict

from astropy.extern import six


__all__ = ['dict_split', 'dict_merge']


MERGE_FOLD_FUNCS = {
    'first':    lambda new, old: old,
    'last':     lambda new, old: new,
    'lowest':   min,
    'highest':  max,
    'shortest': lambda new, old: old if len(old) < len(new) else new,
    'longest':  lambda new, old: new if len(old) < len(new) else old,
    }


def dict_split(dictionary, sep='_'):
    """Splits a `dict` into a dictionary of dictionaries based on the  \
            :meth:`str.split` keys.

    Parameters
    ----------
    dictionary : `dict`-like
        The dictionary that should be split

    sep : `str`, optional
        The :meth:`str.split` seperator.
        Default is ``"_"``.

    Returns
    -------
    dict of dicts : `collections.defaultdict`
        The dictionary of dictionaries that resulted from splitting the
        original dictionary.

    Raises
    ------
    ValueError
        If any key doesn't contain the sep and cannot be split into 2
        substrings.

    Examples
    --------
    A simple example::

        >>> from nddata.utils.dictutils import dict_split
        >>> a_dictionary = {'a_a': 10, 'a_b':15, 'b_b': 20, 'b_a': 10}

        >>> # Actually no need to give the seperator explicitly here:
        >>> dict_of_dicts = dict_split(a_dictionary, '_')
        >>> dict_of_dicts  # doctest: +SKIP
        defaultdict(dict, {'a': {'a': 10, 'b': 15}, 'b': {'a': 10, 'b': 20}})

    The `~collections.defaultdict` can be used like a normal `dict` except that
    it raises no KeyError when trying to get or set a key that doesn't exist
    in the dictionary. If you feel uncomfortable with the defaultdict, just
    cast it to a normal dictionary afterwards::

        >>> a_dictionary = {'a_a': 10, 'a_b':15, 'b_b': 20, 'b_a': 10}
        >>> dict_of_dicts = dict_split(a_dictionary)
        >>> dict(dict_of_dicts)  # doctest: +SKIP
        {'a': {'a': 10, 'b': 15}, 'b': {'a': 10, 'b': 20}}

    Also other seperators are possible, for example a whitespace::

        >>> a_dictionary = {'a a': 10, 'a b':15, 'b b': 20, 'b a': 10}
        >>> dict_of_dicts = dict_split(a_dictionary, ' ')
        >>> dict_of_dicts  # doctest: +SKIP
        defaultdict(dict, {'a': {'a': 10, 'b': 15}, 'b': {'a': 10, 'b': 20}})

    """
    # Create a defaultdictionary so that we don't need any try/except or
    # "if x in dictionary" code. I just hope that the users don't get annoyed
    # if they see an unknown type returned.
    # FIXME: Maybe better to cast it to a dict at the end? Is it worth it?
    res_dict = defaultdict(dict)
    for key in dictionary:  # each key in the input
        # split the key but only once, we don't want lists as indices...
        pre, suf = key.split(sep, 1)
        # Here comes the magic of the defaultdict, we don't need to create new
        # subdicts if the key isn't in the dictionary already so we can just
        # double indix here.
        res_dict[pre][suf] = dictionary[key]
    return res_dict


def dict_merge(dict1, *dicts, **foldfunc):
    """Merge an arbitary number of :class:`dict`-like objects.

    Parameters
    ----------
    *dicts : dict-like
        The dictionaries to be merged. At least two must be given.

    foldfunc : callable, str or None
        The function that determines which value will be kept if two dicts
        contain the same key. The callable must take two arguments, the first
        will be the first found value and the second the new found one.
        ``None`` will just keep the latest found value. There are some strings
        that use a predefined function.

        - **"min"** : keep the lower value
        - **"max"** : keep the higher value
        - **"first"** : keep the first found value
        - **"shortest"** : keep the shorter value
        - **"longest"** : keep the longer value

        And some suggestions for possible callables.

        - `sum` : keep the sum of both values
        - `operator.add` : actually the same as sum but more efficient
        - `operator.mul` : keep the product of the values

        the only requirement is that the callable takes two arguments.

    Returns
    -------
    result : type of the first dictionary
        The merged dictionaries.

    Raises
    ------
    ValueError
        If the ``foldfunc`` was a string that wasn't registered as valid
        synonym for a predefined fold function.

    TypeError
        If the ``foldfunc`` did not take two arguments.

    Examples
    --------
    The examples are using `collections.OrderedDict` so the order of the
    attributes is always fixed, but normal `dict` are also possible.

    To merge multiple dictionaries keeping the last found value in case of
    conflicts::

        >>> from nddata.utils.dictutils import dict_merge
        >>> from collections import OrderedDict

        >>> a = OrderedDict([('a', 2), ('b', 3), ('c', 1)])
        >>> b = OrderedDict([('b', 2), ('c', 2), ('d', 2)])
        >>> merged = dict_merge(a, b)

        >>> merged
        OrderedDict([('a', 2), ('b', 2), ('c', 2), ('d', 2)])

    To get the smallest value in case of conflicts::

        >>> merged = dict_merge(a, b, foldfunc=min)
        >>> merged
        OrderedDict([('a', 2), ('b', 2), ('c', 1), ('d', 2)])

    To keep the first found value you don't need to create an own function,
    simply give ``"first"`` as fold function::

        >>> merged = dict_merge(a, b, foldfunc='first')
        >>> merged
        OrderedDict([('a', 2), ('b', 3), ('c', 1), ('d', 2)])

    More than two dictionaries are possible as well::

        >>> a = OrderedDict([('a', 'a'), ('b', 'bb'), ('c', 'cc')])
        >>> b = OrderedDict([('b', 'b'), ('c', 'ccc'), ('d', 'd')])
        >>> c = OrderedDict([('a', 'aa'), ('b', 'b'), ('c', 'cc')])
        >>> d = OrderedDict([('b', 'bbb'), ('c', 'ccccc'), ('d', 'dd')])

        >>> merged = dict_merge(a, b, c, d, foldfunc='longest')
        >>> merged
        OrderedDict([('a', 'aa'), ('b', 'bbb'), ('c', 'ccccc'), ('d', 'dd')])
    """
    # There is unfortunatly no keyword-only possibility for python2 so just use
    # a kwargs-like approach
    foldfunc = foldfunc.get('foldfunc', None)
    # Copy the first dict so the result's class and its properties are defined
    result = dict1.copy()

    if foldfunc is None:
        # Update the result with the contents of the other dicts. This will
        # overwrite every already existing key. So the last encountered value
        # for each key is present in the result.
        for d in dicts:
            result.update(d)
        return result
    elif isinstance(foldfunc, six.string_types):
        foldfunc = MERGE_FOLD_FUNCS.get(foldfunc, None)
        if foldfunc is None:
            raise ValueError('unrecognized fold function.')

    # Now iterate over each dictionary since there is no directly useable
    # dict-method for this kind of operation
    for d in dicts:
        # Now iterate over each key of this dict.
        # This way is faster than "for kw in d.keys()".
        for kw in d:
            # One could also use "try ... except KeyError ..." here instead of
            # the "if kw in result". That would be a bit faster if all dicts
            # contained mostly the same keys ... but since contain checks with
            # dictionaries are relativly cheap it doesn't make a huge
            # difference
            if kw in result:
                # The key was already present in the result so apply the fold
                # function.
                result[kw] = foldfunc(d[kw], result[kw])

            else:
                # If the key was not yet in the result dict wrap the value
                # inside a list before setting it in the dict.
                result[kw] = d[kw]

    return result
