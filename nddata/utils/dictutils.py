# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from collections import defaultdict

from astropy.extern import six


__all__ = ['dict_split', 'dict_merge', 'dict_merge_keep_all',
           'dict_merge_keep_all_fill_missing']


MERGE_FOLD_FUNCS = {
    'first': lambda new, old: old,
    'last': lambda new, old: new,
    'lowest': min,
    'highest': max,
    'shortest': lambda new, old: old if len(old) < len(new) else new,
    'longest': lambda new, old: new if len(old) < len(new) else old,
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
    res_dict = defaultdict(dict)
    for key in dictionary:  # each key in the input
        # split the key but only once, we don't want lists as indices...
        pre, suf = key.split(sep, 1)
        # Here comes the magic of the defaultdict, we don't need to create new
        # subdicts if the key isn't in the dictionary already so we can just
        # double indix here.
        res_dict[pre][suf] = dictionary[key]
    return res_dict


def dict_merge(*dicts, **foldfunc):
    """Merge an arbitary number of :class:`dict`-like objects.

    Parameters
    ----------
    *dicts : `dict`-like
        The dictionaries to be merged.

    foldfunc : callable, `str` or None, optional
        The function that determines which value will be kept if two dicts
        contain the same key. The callable must take two arguments, the first
        will be the first found value and the second the new found one.
        ``None`` will just keep the last found value. There are also some
        strings that use a predefined function.

        - **"min"** : keep the lowest value
        - **"max"** : keep the highest value
        - **"first"** : keep the first found value
        - **"shortest"** : keep the shorter value
        - **"longest"** : keep the longer value

        And some suggestions for possible callables.

        - `sum` : keep the sum of both values
        - `operator.add` : actually the same as sum but more efficient
        - `operator.mul` : keep the product of the values

        The only requirement is that the callable takes two arguments.
        Default is ``None.

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

        >>> dict_merge(a, b)
        OrderedDict([('a', 2), ('b', 2), ('c', 2), ('d', 2)])

    To get the smallest value in case of conflicts::

        >>> dict_merge(a, b, foldfunc=min)
        OrderedDict([('a', 2), ('b', 2), ('c', 1), ('d', 2)])

    To keep the first found value you don't need to create an own function,
    simply give ``"first"`` as fold function::

        >>> dict_merge(a, b, foldfunc='first')
        OrderedDict([('a', 2), ('b', 3), ('c', 1), ('d', 2)])

    More than two dictionaries are possible as well::

        >>> a = OrderedDict([('a', 'a'), ('b', 'bb'), ('c', 'cc')])
        >>> b = OrderedDict([('b', 'b'), ('c', 'ccc'), ('d', 'd')])
        >>> c = OrderedDict([('a', 'aa'), ('b', 'b'), ('c', 'cc')])
        >>> d = OrderedDict([('b', 'bbb'), ('c', 'ccccc'), ('d', 'dd')])

        >>> dict_merge(a, b, c, d, foldfunc='longest')
        OrderedDict([('a', 'aa'), ('b', 'bbb'), ('c', 'ccccc'), ('d', 'dd')])
    """
    # If no dicts were given just return an empty dictionary.
    if not dicts:
        return {}

    # TODO: As soon as python2 isn't supported anymore the definition could be
    # changed to dict_merge(dict1, *dicts, foldfunc=None) to make foldfunc a
    # keyword only argument.
    # But since that happens we assume that the foldfunc is given explicitly by
    # name and we can get it out of the kwargs (in this case called foldfunc)
    # I just hope that is not confusing. The following line will discard any
    # kwargs except foldfunc and if that's not in it it defaults to "None".
    foldfunc = foldfunc.get('foldfunc', None)

    # The first dict determines the class of the result so we copy it. I
    # deliberatly choose not to deepcopy here since all the other attributes
    # will also be simply assigned. The user should be aware that it is only a
    # shallow copy.
    result = dicts[0].copy()

    # The easy way: There was no foldfunc or it was set to None. This is just
    # updating the dicts and keeping the last encountered key.
    if foldfunc is None:
        # Just use the method "update" for each of the other dictionaries and
        # we are done.
        for d in dicts:
            result.update(d)
        return result

    # The other case: It is a string, so we check if the string is registered
    # and has a predefined foldfunction already. (See top of this module which
    # are defined). In case we find no match raise a ValueError.
    elif isinstance(foldfunc, six.string_types):
        foldfunc = MERGE_FOLD_FUNCS.get(foldfunc, None)
        if foldfunc is None:
            raise ValueError('unrecognized fold function.')

    # Now iterate over each dictionary since there is no directly useable
    # dict-method for this kind of operation.
    for d in dicts[1:]:
        # Now iterate over each key of this dict.
        # This way is faster than "for kw in d.keys()".
        for kw in d:
            # One could also use "try ... except KeyError ..." here instead of
            # the "if kw in result". That would be a bit faster if all dicts
            # contained mostly the same keys ... but since contain checks with
            # dictionaries are relativly cheap: it doesn't make a huge
            # difference
            if kw in result:
                # The key was already present in the result so apply the fold
                # function.
                result[kw] = foldfunc(d[kw], result[kw])

            else:
                # If the key was not yet in the result dict just create a new
                # key-value pair.
                result[kw] = d[kw]

    return result


def dict_merge_keep_all(*dicts):
    """Merge an arbitary number of :class:`dict`-like objects while keeping \
            all encountered values.

    Each key of the result will contain a list containing all found values for
    this key.

    Parameters
    ----------
    *dicts : `dict`-like
        The dictionaries to be merged.

    Returns
    -------
    result : type of the first dictionary
        The merged dictionaries.

    Examples
    --------
    These examples work with `collections.OrderedDict` which is a `dict` that
    keeps the order in which the keys were inserted. This is primarly because
    of the doctests failing otherwise but also to show that the kind of `dict`
    doesn't matter because it is kept during the merge::

        >>> from collections import OrderedDict
        >>> from nddata.utils.dictutils import dict_merge_keep_all

        >>> a = OrderedDict([('a', 1), ('b', 1)])
        >>> b = OrderedDict([('b', 2), ('c', 2)])

        >>> dict_merge_keep_all(a, b)
        OrderedDict([('a', [1]), ('b', [1, 2]), ('c', [2])])
    """
    # If no dicts were given just return an empty dictionary.
    if not dicts:
        return {}

    # Much of it is identical to the base dict_merge: First we copy the first
    # dictionary to return the same type of dict-class.
    result = dicts[0].copy()

    # Iterate once over the key-value pairs already inserted and wrap the
    # values in a list so we can append to these lists.
    for kw in result:
        result[kw] = [result[kw]]

    # Now iterate over each dictionary
    for d in dicts[1:]:
        # Now iterate over each key of this dict.
        for kw in d:
            if kw in result:
                # The key was already present in the result so just append it
                # to the list.
                result[kw].append(d[kw])
            else:
                # If the key was not yet in the result dict wrap the value
                # inside a list before setting it in the dict.
                result[kw] = [d[kw]]

    return result


def dict_merge_keep_all_fill_missing(*dicts, **fill):
    """Merge an arbitary number of :class:`dict`-like objects while keeping \
            all encountered values and replace missing.

    Each key of the result will contain a list with the size of the number of
    merged dictionaries. The i-th element will correspond to the value of the
    i-th dictionary of contain a fill value in case that dictionary didn't have
    this key.

    Parameters
    ----------
    *dicts : dict-like
        The dictionaries to be merged.

    fill : any type, optional
        Fill missing values with this value. A missing value is if one of the
        dictionaries does not have this key.
        Default is ``0``.

    Returns
    -------
    result : any type
        The merged dictionaries.

    Notes
    -----
    The first given ``dict`` is copied to provide the class and
    properties of the result.

    Examples
    --------
    These examples work with `collections.OrderedDict` which is a `dict` that
    keeps the order in which the keys were inserted. This is primarly because
    of the doctests failing otherwise but also to show that the kind of `dict`
    doesn't matter because it is kept during the merge::

        >>> from collections import OrderedDict
        >>> from nddata.utils.dictutils import dict_merge_keep_all_fill_missing

        >>> a = OrderedDict([('a', 1), ('b', 1)])
        >>> b = OrderedDict([('b', 2), ('c', 2)])

        >>> dict_merge_keep_all_fill_missing(a, b)
        OrderedDict([('a', [1, None]), ('b', [1, 2]), ('c', [None, 2])])

    or with a differing ``fill`` value::

        >>> dict_merge_keep_all_fill_missing(a, b, fill=0)
        OrderedDict([('a', [1, 0]), ('b', [1, 2]), ('c', [0, 2])])
    """
    # If no dicts were given just return an empty dictionary.
    if not dicts:
        return {}

    # TODO: As soon as python2 isn't supported anymore the definition could be
    # changed to dict_merge_keep_all_fill_missing(dict1, *dicts, fill=0). See
    # dict_merge for a similar discussion on keyword-only arguments in python2.
    fill = fill.get('fill', None)

    # Copy the first dict so the result's class and its properties are defined
    result = dicts[0].copy()

    # The number of dictionaries so we know what length of lists we need as
    # stubs.
    n_dicts = len(dicts)

    # Wrap every value of the first dict in a list and append the placeholders
    # Because of the addition these will create new lists so we don't need to
    # copy here.
    for kw in result:
        result[kw] = [result[kw]] + [fill] * (n_dicts - 1)

    # Now go through the other dicts but start the enumeration with 1 otherwise
    # we would overwrite the entries from the first dictionary.
    for idx, d in enumerate(dicts[1:], 1):
        for kw in d:
            if kw not in result:
                result[kw] = [fill] * n_dicts
            # Insert it at the right position in any case
            result[kw][idx] = d[kw]

    return result
