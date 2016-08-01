# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from collections import defaultdict, ItemsView, KeysView, ValuesView, Mapping

from astropy.extern import six

from .itertools_recipes import unique_everseen
from .sentinels import ParameterNotSpecified


__all__ = ['dict_split', 'dict_merge', 'dict_merge_keep_all',
           'dict_merge_keep_all_fill_missing', 'ListDict']


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
        Default is ``None``.

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
        If the ``foldfunc`` doesn't take two arguments.

    Notes
    -----
    The first given ``dict`` is copied to provide the class and
    properties of the result.

    .. note::
        Known Bugs:

        - `astropy.io.fits.Header` may cause problems if it contains multiple
          values for one key, for example multiple ``"COMMENT"`` cards.

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

        # Another approach would be to use the update and create intermediate
        # generators with the appropriate values:
        # result.update((k, foldfunc(v, result[k])) if k in result else (k, v)
        #               for k, v in d.items())
        # or to stick with the iterations I used originally:
        # result.update((kw, foldfunc(d[kw], result[kw])) if kw in result else
        #               (kw, d[kw])
        #               for kw in d)
        # but in the most common cases this was almost a factor 2-3 slower and
        # a bit less readable so I dismissed it. No idea why it is really
        # slower...

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

    Notes
    -----
    .. note::
        Known Bugs:

        - `astropy.io.fits.Header` cannot be used as class of the first
          argument because it doesn't support `list` as value.

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

    .. note::
        Because the list size cannot be determined before the iteration this
        function is slower than `dict_merge_keep_all_fill_missing`.
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

    .. note::
        Known Bugs:

        - `astropy.io.fits.Header` cannot be used as class of the first
          argument because it doesn't support `list` as value.

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


class ListDict(object):
    """Simulates a dictionary that contains the values of
    multiple dictionaries for each key.

    Parameters
    ----------
    dictionaries : `dict`-like
        An arbitary number of dictionary-like objects.

    fill : any type, optional
        If the key was not present in any of the dictionaries
        this fill values is assumed instead.
        Default is ``None``.

    Examples
    --------
    For example two dictionaries::

        >>> from nddata.utils.dictutils import ListDict
        >>> d1 = {'a': 10, 'b': 10, 'c': 10}
        >>> d2 = {'a': 20, 'b': 20, 'd': 20}
        >>> ld = ListDict(d1, d2)
        >>> ld
        ListDict (4 distinct keys in 2 objects)

    Adding more dictionaries can be done with :meth:`update` or the **inplace**
    ``+=`` operator::

        >>> d3 = {'a': 30, 'b': 30, 'e': 30}
        >>> ld.update(d3)
        >>> ld
        ListDict (5 distinct keys in 3 objects)

        >>> d4 = {'a': 40, 'b': 40, 'f': 40}
        >>> ld += d4
        >>> ld
        ListDict (6 distinct keys in 4 objects)

    The instance can be accessed like a normal dictionary. Either with
    direct indexing::

        >>> ld['a']
        (10, 20, 30, 40)

    Or by using :meth:`keys`, :meth:`values` or :meth:`items`::

        >>> sorted(ld.keys())
        ['a', 'b', 'c', 'd', 'e', 'f']

        >>> sorted(ld.items())
        [('a', (10, 20, 30, 40)),
         ('b', (10, 20, 30, 40)),
         ('c', (10, None, None, None)),
         ('d', (None, 20, None, None)),
         ('e', (None, None, 30, None)),
         ('f', (None, None, None, 40))]

         >>> (10, None, None, None) in ld.values()
         True

    These can be used like the equivalent dictionary methods. For example as
    iterators or for ``in`` checks. However the order may differ if the
    saved dictionaries are not ordered.

    The `fill` value defaults to ``None`` but can be set when creating the
    instance or by setting the property::

        >>> ld = ListDict(d1, d2, d3, d4, fill=0)
        >>> ld['c']
        (10, 0, 0, 0)

        >>> ld.fill = '--'
        >>> ld['c']
        (10, '--', '--', '--')

    Or just temporarly set by using :meth:`get`::

        >>> ld.get('c', 'x')
        (10, 'x', 'x', 'x')

        >>> ld.get('c')  # without a fill value this is equivalent to "['c']"
        (10, '--', '--', '--')

    Accessing a key that is not present in any of the dictionaries
    **does not** raise a ``KeyError``::

        >>> ld['q']
        ('--', '--', '--', '--')

    If one wants to check if a key is set in any of the dictionaries use
    ``in``::

        >>> 'q' in ld
        False
        >>> 'e' in ld
        True

    Changes to the original dictionaries are propagated to the `ListDict`::

        >>> d4['c'] = 40
        >>> ld['c']
        (10, '--', '--', 40)

    .. note::
        The values of the dictionaries cannot be changed through `ListDict`.

    .. note::
        The values are created on the fly. Using many and/or large
        dictionaries can be quite slow.

    `repr` and `str` of the instance are identical::

        >>> ld
        ListDict (6 distinct keys in 4 objects)
        >>> str(ld)
        'ListDict (6 distinct keys in 4 objects)'

    And can be compared like normal dictionaries (the order of the keys
    does not matter)::

        >>> ListDict(d1, d2, d3, d4) == ListDict(d1, d2, d3, d4)
        True

    But the order of the dictionaries matters::

        >>> ListDict(d1, d2) != ListDict(d2, d1)
        True

    But also the fill value does matter::

        >>> ListDict(d1, d2, fill=0) == ListDict(d1, d2, fill=1)
        False
    """
    def __init__(self, *dictionaries, **kwargs):
        # Always convert the tuple of dictionaries to a list of
        # these.
        if not dictionaries:
            self._dictionaries = []
        else:
            self._dictionaries = list(dictionaries)

        # Process keyword arguments (python2 cannot handle keyword-only args)
        self.fill = kwargs.pop('fill', None)
        if kwargs:
            msg = ['{0}={1}'.format(key, kwargs[key]) for key in kwargs]
            raise TypeError('Unrecognized argument(s) "{0}".'
                            ''.format(', '.join(msg)))

    @property
    def fill(self):
        """(any type) Fill value for missing entries.
        """
        return self._fill

    @fill.setter
    def fill(self, value):
        self._fill = value

    @property
    def ndicts(self):
        """(int) Number of saved dictionaries (readonly).
        """
        return len(self._dictionaries)

    @property
    def nkeys(self):
        """(int) Number of keys in the dictionaries (readonly).

        This is equivalent to calling `len` on the instance.
        """
        return len(self)

    def keys(self):
        """Get the keys of the dictionaries.

        Returns
        -------
        values : `collections.KeysView`
            A generator containing the keys of the dictionaries.

        Notes
        -----
        The keys are unordered except the keys of the dictionaries
        themselves are ordered.
        """
        return KeysView(self)

    def values(self):
        """Get the values of the dictionaries.

        Returns
        -------
        values : `collections.ValuesView`
            A generator containing the values of the dictionaries.

        Notes
        -----
        The values are unordered with respect to the keys except the
        dictionaries themselves are ordered.
        """
        return ValuesView(self)

    def items(self):
        """Get the (key, value) pairs of the dictionaries.

        Returns
        -------
        items : `collections.ItemsView`
            A generator containing the (key, value) pairs of the
            dictionaries.

        Notes
        -----
        The keys are unordered except the dictionaries themselves
        are ordered.
        """
        return ItemsView(self)

    def __repr__(self):
        return ('{0.__class__.__name__} ({0.nkeys} distinct keys in {0.ndicts}'
                ' objects)'.format(self))
        # cls_name = self.__class__.__name__
        # indentation = len(cls_name) + 1
        # linebreak = ', \n{0}'.format(' ' * indentation)
        # entries = ['{0}: {1}'.format(key, self[value]) for key in self]
        # return '{0}({1})'.format(cls_name, linebreak.join(entries))

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return sum(1 for _ in self)

    def __contains__(self, key):
        return any(key in dct for dct in self._dictionaries)

    def __iter__(self):
        return unique_everseen(key for dictionary in self._dictionaries
                               for key in dictionary)

    def __getitem__(self, key):
        return self.get(key, self.fill)

    def get(self, key, fill=ParameterNotSpecified):
        """Get the values for ``key`` with the option to temporaryly
        overwrite the `fill` value.

        Parameters
        ----------
        key : any type
            Get the values for this key.

        fill : any type, optional
            If given use this value instead of `fill`, otherwise `fill` is
            used.

        Returns
        -------
        values : `tuple` of values
            The values for this key of each dictionary as tuple.
        """
        if fill is ParameterNotSpecified:
            fill = self.fill
        return tuple(dct.get(key, fill) for dct in self._dictionaries)

    def update(self, dictionary):
        """Add a dictionary to the list of dictionaries.

        Parameters
        ----------
        dictionary : `collections.Mapping`
            The mapping to append.
        """
        self._dictionaries.append(dictionary)

    def __iadd__(self, dictionary):
        self.update(dictionary)
        return self

    def __eq__(self, other):
        return dict(self.items()) == dict(other.items())

    def __ne__(self, other):
        return not self == other

# Register it as a Mapping to allow "duck typing".
Mapping.register(ListDict)
