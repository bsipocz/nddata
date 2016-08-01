# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function)
# TODO: Unicode literals would fail the tests here. Since internally these
# functions don't use strings we can just ignore it here!

from itertools import (islice, count, groupby, chain, repeat, starmap, tee,
                       cycle, combinations)
from collections import deque
from operator import mul, itemgetter
from random import choice, sample, randrange
from copy import copy

from astropy.extern.six.moves import zip_longest, filterfalse
from astropy.extern import six

if six.PY2:  # pragma: no cover
    from future_builtins import zip, map, filter
    range = xrange


__all__ = ['take', 'tabulate', 'tail', 'consume', 'nth', 'all_equal',
           'quantify', 'padnone', 'ncycles', 'dotproduct', 'flatten',
           'repeatfunc', 'pairwise', 'grouper', 'roundrobin', 'powerset',
           'unique_everseen', 'unique_justseen', 'iter_except', 'first_true',
           'random_product', 'random_permutation', 'random_combination',
           'random_combination_with_replacement', 'tee_lookahead', 'last_true']

# FIXME: Due to a change in random.choice the results are not reproducible for
# between pre python 3.2 and afterwards.
if six.PY3:  # pragma: no cover
    __doctest_skip__ = ['random_combination_with_replacement',
                        'random_permutation', 'random_product']


def take(iterable, n):
    """Return first n items of the iterable as a list.

    Parameters
    ----------
    iterable : `collections.Iterable`
        Any iterable from which to take the items.

    n : `int`
        Number of items to take from the iterable.

    Returns
    -------
    items : `list`
        The first items of the iterable.

    Examples
    --------
    >>> from nddata.utils.itertools_recipes import take
    >>> take(range(10, 20), 5)
    [10, 11, 12, 13, 14]
    """
    return list(islice(iterable, n))


def tabulate(function, start=0):
    """Return function(0), function(1), ...

    Parameters
    ----------
    function : `collections.Callable`
        The function to apply.

    start : `int`, optional
        The starting value to apply the function on. For each `next` call it
        will be incremented by one.

    Returns
    -------
    tabulated : generator
        An infinite generator containing the results of the function applied on
        the values beginning by start.

    Examples
    --------
    Since the return is a never stopping generator you need some other function
    to extract only the needed values. For example `take`::

        >>> from nddata.utils.itertools_recipes import tabulate, take
        >>> from math import sqrt
        >>> t = tabulate(sqrt, 0)
        >>> take(t, 3)
        [0.0, 1.0, 1.4142135623730951]

    .. warning::
        This will return an infinitly long generator so do **not** try to do
        something like ``list(tabulate())``!
    """
    return map(function, count(start))


def tail(iterable, n):
    """Return an iterator over the last n items.

    Parameters
    ----------
    iterable : `collections.Iterable`
        The iterable from which to take the last items.

    n : `int`
        How many elements.

    Returns
    -------
    iterator : `collections.Iterator`
        The last n items as iterator.

    Examples
    --------
    >>> from nddata.utils.itertools_recipes import tail
    >>> list(tail('ABCDEFG', 3))
    ['E', 'F', 'G']
    """
    # tail(3, 'ABCDEFG') --> E F G
    return iter(deque(iterable, maxlen=n))


def consume(iterator, n):
    """Advance the iterator n-steps ahead. If n is none, consume entirely.

    Parameters
    ----------
    iterator : `collections.Iterator`
        Any iterator from which to consume the items.

    n : `int`, `None`
        Number of items to consume from the iterable. If ``None`` consume it
        entirely.

    Returns
    -------
    consumed_iterator : `list`
        The iterator without the consumed items.

    Examples
    --------
    >>> from nddata.utils.itertools_recipes import consume
    >>> g = (x**2 for x in range(10))
    >>> consume(g, 2)
    >>> list(g)
    [4, 9, 16, 25, 36, 49, 64, 81]

    >>> g = (x**2 for x in range(10))
    >>> consume(g, None)
    >>> list(g)
    []
    """
    # Use functions that consume iterators at C speed.
    if n is None:
        # feed the entire iterator into a zero-length deque
        deque(iterator, maxlen=0)
    else:
        # advance to the empty slice starting at position n
        next(islice(iterator, n, n), None)


def nth(iterable, n, default=None):
    """Returns the nth item or a default value.

    Parameters
    ----------
    iterable : `collections.Iterable`
        Any iterable from which to take the item.

    n : `int`
        Index of the item.

    default : any type, optional
        Default value if the iterable doesn't contain the index.
        Default is ``None``.

    Returns
    -------
    nth_item : any type
        The nth item of the iterable or default if the index wasn't present in
        the iterable.

    Examples
    --------
    Without default value::

        >>> from nddata.utils.itertools_recipes import nth
        >>> g = (x**2 for x in range(10))
        >>> nth(g, 5)
        25

    Or with default if the index is not present::

        >>> g = (x**2 for x in range(10))
        >>> nth(g, 15, 0)
        0
    """
    return next(islice(iterable, n, None), default)


def all_equal(iterable):
    """Returns True if all the elements are equal to each other.

    Parameters
    ----------
    iterable : `collections.Iterable`
        Any iterable to test.

    Returns
    -------
    all_equal : `bool`
        True if all elements are equal or False if not.

    Examples
    --------
    >>> from nddata.utils.itertools_recipes import all_equal
    >>> all_equal([1,1,1,1,1,1,1,1,1])
    True

    >>> all_equal([1,1,1,1,1,1,1,2,1])
    False
    """
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def quantify(iterable, pred=bool):
    """Count how many times the predicate is true.

    Parameters
    ----------
    iterable : `collections.Iterable`
        Any iterable to count in.

    pred : `collections.Callable`, optional
        Predicate to test.
        Default is ``bool``.

    Returns
    -------
    number : number
        The numer of times the predicate is True.

    Examples
    --------
    >>> from nddata.utils.itertools_recipes import quantify
    >>> quantify([0,0,'',{}, [], 2])
    1

    >>> def smaller5(val): return val < 5
    >>> quantify([1,2,3,4,5,6,6,7], smaller5)
    4
    """
    return sum(map(pred, iterable))


def padnone(iterable):
    """Returns the sequence elements and then returns None indefinitely.

    Useful for emulating the behavior of the :func:`map` function.

    Parameters
    ----------
    iterable : `collections.Iterable`
        Any iterable to pad.

    Returns
    -------
    generator : generator
        A generator containing the iterable followed by infinitely many None.

    Examples
    --------
    >>> from nddata.utils.itertools_recipes import padnone
    >>> take(padnone([1,2,3]), 5)
    [1, 2, 3, None, None]

    .. warning::
        This will return an infinitly long generator so do **not** try to do
        something like ``list(padnone())``!
    """
    return chain(iterable, repeat(None))


def ncycles(iterable, n):
    """Returns the sequence elements n times.

    Parameters
    ----------
    iterable : `collections.Iterable`
        Any iterable to repeat.

    n : `int`
        Number of repeatitions.

    Returns
    -------
    repeated_iterable : generator
        The iterable repeated n times.

    Examples
    --------
    >>> from nddata.utils.itertools_recipes import ncycles
    >>> list(ncycles([1,2,3], 3))
    [1, 2, 3, 1, 2, 3, 1, 2, 3]
    """
    return chain.from_iterable(repeat(tuple(iterable), n))


def dotproduct(vec1, vec2):
    """Dot product (matrix multiplication) of two vectors.

    Parameters
    ----------
    vec1, vec2 : `collections.Iterable`
        Any iterables to calculate the dot product.

    Returns
    -------
    dotproduct : number
        The dot product - the sum of the element-wise multiplication.

    Examples
    --------
    >>> from nddata.utils.itertools_recipes import dotproduct
    >>> dotproduct([1,2,3,4], [1,2,3,4])
    30
    """
    return sum(map(mul, vec1, vec2))


def flatten(iterable):
    """Flatten one level of nesting.

    Parameters
    ----------
    iterable : `collections.Iterable`
        Any iterable to flatten.

    Returns
    -------
    flattened_iterable : generator
        The iterable with the first level of nesting flattened.

    Examples
    --------
    >>> from nddata.utils.itertools_recipes import flatten
    >>> list(flatten([[1,2,3,4], [4,3,2,1]]))
    [1, 2, 3, 4, 4, 3, 2, 1]
    """
    return chain.from_iterable(iterable)


def repeatfunc(func, *args, **times):
    """Repeat calls to func with specified arguments.

    Parameters
    ----------
    func : `collections.Callable`
        The function that will be called.

    args :
        optional arguments for the ``func``.

    times : `int`, `None`, optional
        The number of times the function is called. If ``None`` there will be
        no limit.
        Default is ``None``.

    Returns
    -------
    iterable : generator
        The result of the repeatedly called function.

    Examples
    --------
    >>> from nddata.utils.itertools_recipes import repeatfunc
    >>> import random

    >>> random.seed(5)
    >>> take(repeatfunc(random.random), 5)
    [0.6229016948897019,
     0.7417869892607294,
     0.7951935655656966,
     0.9424502837770503,
     0.7398985747399307]

    >>> random.seed(2)
    >>> list(repeatfunc(random.random, times=3))
    [0.9560342718892494, 0.9478274870593494, 0.05655136772680869]
    >>> random.seed(None)

    .. warning::
        This will return an infinitly long generator if you don't specify
        ``times``.
    """
    times = times.get('times', None)
    if times is None:
        return starmap(func, repeat(args))
    return starmap(func, repeat(args, times))


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ...

    Parameters
    ----------
    iterable : `collections.Iterable`
        Any iterable to pairwise combine.

    Returns
    -------
    pairwise : generator
        An iterable containing tuples of sucessive elements of the iterable.

    Examples
    --------
    >>> from nddata.utils.itertools_recipes import pairwise
    >>> list(pairwise([1,2,3]))
    [(1, 2), (2, 3)]
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def grouper(iterable, n, fillvalue=None):
    """Collect data into fixed-length chunks or blocks.

    Parameters
    ----------
    iterable : `collections.Iterable`
        Any iterable to group.

    n : `int`
        The number of groups/chunks

    fillvalue : any type, optional
        The fillvalue if one group is not yet filled but the iterable is
        consumed.
        Default is ``None``.

    Returns
    -------
    groups : generator
        An iterable containing the groups/chunks as `tuple`.

    Examples
    --------
    >>> from nddata.utils.itertools_recipes import grouper
    >>> list(grouper('ABCDEFG', 3, 'x'))
    [('A', 'B', 'C'), ('D', 'E', 'F'), ('G', 'x', 'x')]
    """
    args = [iter(iterable)] * n
    if six.PY2:
        return zip_longest(fillvalue=fillvalue, *args)
    else:  # pragma: no cover
        return zip_longest(*args, fillvalue=fillvalue)


def roundrobin(*iterables):
    """...

    Parameters
    ----------
    iterables : `collections.Iterable`
        Iterables to combine using the round-robin.

    Returns
    -------
    roundrobin : generator
        An iteable filled with the values of the iterables.

    Examples
    --------
    >>> from nddata.utils.itertools_recipes import roundrobin
    >>> list(roundrobin('ABC', 'D', 'EF'))
    ['A', 'D', 'E', 'B', 'F', 'C']
    """
    # Recipe credited to George Sakkis
    pending = len(iterables)
    if six.PY2:
        nexts = cycle(iter(it).next for it in iterables)
    else:  # pragma: no cover
        nexts = cycle(iter(it).__next__ for it in iterables)
    while pending:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            pending -= 1
            nexts = cycle(islice(nexts, pending))


def partition(iterable, pred):
    """Use a predicate to partition entries into false entries and true \
            entries.

    Parameters
    ----------
    iterable : `collections.Iterable`
        Iterable to partition.

    pred : `collections.Callable`
        The predicate which determines the group in which the value of the
        iterable belongs.

    Returns
    -------
    false_values : generator
        An iterable containing the values for which the predicate was True.

    true_values : generator
        An iterable containing the values for which the predicate was True.

    Examples
    --------
    >>> from nddata.utils.itertools_recipes import partition
    >>> def is_odd(val): return val % 2
    >>> [list(i) for i in partition(range(10), is_odd)]
    [[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]]
    """
    t1, t2 = tee(iterable)
    return filterfalse(pred, t1), filter(pred, t2)


def powerset(iterable):
    """Create all possible sets of values from an iterable.

    Parameters
    ----------
    iterable : `collections.Iterable`
        Iterables for which to create a powerset.

    Returns
    -------
    powerset : generator
        An iterable containing all powersets as tuple.

    Examples
    --------
    >>> from nddata.utils.itertools_recipes import powerset
    >>> list(powerset([1,2,3]))
    [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def unique_everseen(iterable, key=None):
    """List unique elements, preserving order. Remember all elements ever seen.

    Parameters
    ----------
    iterable : `collections.Iterable`
        Iterables to check.

    key : `collections.Callable`, `None`, optional
        If ``None`` the values are taken as they are. If it's a callable the
        callable is applied to the value before comparing it.
        Default is ``None``.

    returns
    -------
    iterable : generator
        An iterable containing all unique values ever seen in the iterable.

    Examples
    --------
    >>> from nddata.utils.itertools_recipes import unique_everseen
    >>> list(unique_everseen('AAAABBBCCDAABBB'))
    ['A', 'B', 'C', 'D']

    >>> list(unique_everseen('ABBCcAD', str.lower))
    ['A', 'B', 'C', 'D']
    """
    seen = set()
    seen_add = seen.add
    if key is None:
        for element in filterfalse(seen.__contains__, iterable):
            seen_add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen_add(k)
                yield element


def unique_justseen(iterable, key=None):
    """List unique elements, preserving order. Remember only the element just \
            seen.

    Parameters
    ----------
    iterable : `collections.Iterable`
        Iterables to check.

    key : `collections.Callable`, `None`, optional
        If ``None`` the values are taken as they are. If it's a callable the
        callable is applied to the value before comparing it.
        Default is ``None``.

    Returns
    -------
    iterable : generator
        An iterable containing all unique values just seen in the iterable.

    Examples
    --------
    >>> from nddata.utils.itertools_recipes import unique_justseen
    >>> list(unique_justseen('AAAABBBCCDAABBB'))
    ['A', 'B', 'C', 'D', 'A', 'B']

    >>> list(unique_justseen('ABBCcAD', str.lower))
    ['A', 'B', 'C', 'A', 'D']
    """
    return map(next, map(itemgetter(1), groupby(iterable, key)))


def iter_except(func, exception, first=None):
    """Call a function repeatedly until an exception is raised.

    Converts a call-until-exception interface to an iterator interface.
    Like ``__builtin__.iter(func, sentinel)`` but uses an exception instead
    of a sentinel to end the loop.

    Parameters
    ----------
    func : `collections.Callable`
        The function that is called until ``exception`` is raised.

    exception : Exception
        The exception which terminates the iteration.

    first : `collections.Callable`, `None`, optional
        This function (if not ``None``) is called once before the ``func`` is
        executed.

    Returns
    -------
    iterable : generator
        The result of the function calls as generator.

    Examples
    --------
    >>> from nddata.utils.itertools_recipes import iter_except
    >>> from collections import OrderedDict

    >>> d = OrderedDict([('a', 1), ('b', 2)])
    >>> list(iter_except(d.popitem, KeyError))
    [('b', 2), ('a', 1)]

    .. note::
        ``d.items()`` would yield the same result. At least with Python3.

    >>> from math import sqrt
    >>> from astropy.extern import six

    >>> g = (sqrt(i) for i in [5, 4, 3, 2, 1, 0, -1, -2, -3])
    >>> func = g.next if six.PY2 else g.__next__
    >>> def say_go(): return 'go'
    >>> list(iter_except(func, ValueError, say_go))
    ['go', 2.23606797749979, 2.0, 1.7320508075688772, 1.4142135623730951, 1.0\
, 0.0]

    Notes
    -----
    Further examples:

    - ``bsddbiter = iter_except(db.next, bsddb.error, db.first)``
    - ``heapiter = iter_except(functools.partial(heappop, h), IndexError)``
    - ``dictiter = iter_except(d.popitem, KeyError)``
    - ``dequeiter = iter_except(d.popleft, IndexError)``
    - ``queueiter = iter_except(q.get_nowait, Queue.Empty)``
    - ``setiter = iter_except(s.pop, KeyError)``
    """
    try:
        if first is not None:
            yield first()
        while 1:
            yield func()
    except exception:
        pass


def first_true(iterable, default=False, pred=None):
    """Returns the first true value in the iterable or default.

    Parameters
    ----------
    iterable : `collections.Iterable`
        The iterable for which to determine the first true value.

    default : any type, optional
        The default value if no true value was found.

    pred : `collections.Callable` or `None`, optional
        If `None` find the first true value. If not `None` find the first value
        for which ``pred(value)`` is true.
        Default is ``None``.

    Examples
    --------
    >>> from nddata.utils.itertools_recipes import first_true
    >>> first_true([0, '', tuple(), 10])
    10

    >>> first_true([0,2,5,8,10], pred=lambda x: x%2)  # First odd number
    5

    >>> first_true([0,0,0,0])
    False

    >>> first_true([0,0,0,0], default=100)  # default value if no true value.
    100
    """
    return next(filter(pred, iterable), default)


def random_product(*iterables, **repeat):
    """Random selection from ``itertools.product(*args, **kwds)``.

    Parameters
    ----------
    iterables : `collections.Iterable`
        Any amount of iterables from which to form the `itertools.product`.

    repeat : `int`, optional
        The number of random samples.
        Default is ``1``.

    Returns
    -------
    sample : `tuple`
        A tuple containing the random samples.

    Examples
    --------
    Take one random sample::

        >>> from nddata.utils.itertools_recipes import random_product
        >>> import random
        >>> random.seed(70)

        >>> random_product(['a', 'b'], [1, 2], [0.5, 0.25])
        ('b', 1, 0.5)

        >>> # or ('a', 2, 0.25) for Python3.

    Or take multiple samples::

        >>> random.seed(10)

        >>> random_product(['a', 'b'], [1, 2], [0.5, 0.25], repeat=5)
        ('b', 1, 0.25, 'a', 2, 0.25, 'b', 1, 0.25, 'a', 1, 0.25, 'b', 1, 0.25)

    Or ``('a', 2, 0.25, 'a', 1, 0.25, 'b', 2, 0.5, 'a', 2, 0.25, 'a', 1,
    0.25)`` with Python3.

        >>> random.seed(None)
    """
    pools = [tuple(pool) for pool in iterables] * repeat.get('repeat', 1)
    return tuple(choice(pool) for pool in pools)


def random_permutation(iterable, r=None):
    """Random selection from ``itertools.permutations(iterable, r)``.

    Parameters
    ----------
    iterable : `collections.Iterable`
        The iterable to permutate.

    r : `int`, `None`, optional
        The number of elements to permutate. If ``None`` use all elements from
        the iterable.
        Default is ``None``.

    Returns
    -------
    random_permutation : `tuple`
        The randomly chosen permutation.

    Examples
    --------
    One random permutation::

        >>> from nddata.utils.itertools_recipes import random_permutation
        >>> import random
        >>> random.seed(20)

        >>> random_permutation([1,2,3,4,5,6])
        (6, 4, 5, 3, 1, 2)

        >>> # Python3: (6, 2, 3, 4, 1, 5)

    One random permutation using a subset of the iterable (here 3 elements)::

        >>> random.seed(5)

        >>> random_permutation([1,2,3,4,5,6], r=3)
        (4, 6, 5)

        >>> # Python3: (5, 3, 6)

        >>> random.seed(None)
    """
    pool = tuple(iterable)
    r = len(pool) if r is None else r
    return tuple(sample(pool, r))


def random_combination(iterable, r):
    """Random selection from ``itertools.combinations(iterable, r)``.

    Parameters
    ----------
    iterable : `collections.Iterable`
        The iterable to combine.

    r : `int`
        The number of elements to combine.

    Returns
    -------
    random_permutation : `tuple`
        The randomly chosen combination.

    Examples
    --------
    >>> from nddata.utils.itertools_recipes import random_combination
    >>> import random
    >>> random.seed(5)

    >>> random_combination([1,2,3,4,5,6], r=4)
    (3, 4, 5, 6)

    >>> random.seed(None)
    """
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(sample(range(n), r))
    return tuple(pool[i] for i in indices)


def random_combination_with_replacement(iterable, r):
    """Random selection from itertools.combinations_with_replacement(iterable, r).

    Parameters
    ----------
    iterable : `collections.Iterable`
        The iterable to combine.

    r : `int`
        The number of elements to combine.

    Returns
    -------
    random_permutation : `tuple`
        The randomly chosen combination with replacement.

    Examples
    --------
    >>> from nddata.utils.itertools_recipes import random_combination_with_replacement
    >>> import random
    >>> random.seed(100)

    >>> random_combination_with_replacement([1,2,3,4,5,6], r=4)
    (1, 3, 5, 5)

    >>> # For Python3 the result will be (2, 2, 4, 4).
    >>> random.seed(None)
    """
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(randrange(n) for i in range(r))
    return tuple(pool[i] for i in indices)


def tee_lookahead(t, i):
    """Inspect the i-th upcomping value from a tee object while leaving the tee
    object at its current position.

    Parameters
    ----------
    t : tee
        The tee object in which to look ahead.

    i : `int`
        The index counting from the current position which should be peeked.

    Returns
    -------
    peek : any type
        The element at the i-th upcoming index in the tee object.

    Raises
    ------
    IndexError
        If the underlying iterator doesn't have enough values.

    Examples
    --------
    >>> from nddata.utils.itertools_recipes import tee_lookahead
    >>> from itertools import tee
    >>> t1, t2 = tee([1,2,3,4,5,6])
    >>> tee_lookahead(t1, 0)
    1
    >>> tee_lookahead(t1, 1)
    2
    >>> tee_lookahead(t1, 0)
    1
    """
    for value in islice(copy(t), i, None):
        return value
    raise IndexError(i)


# Special ones (not from the python documentation)


def last_true(iterable, default=False, pred=None):
    """Returns the last true value in the iterable or default.

    Parameters
    ----------
    iterable : `collections.Iterable`
        The iterable for which to determine the last true value.

    default : any type, optional
        The default value if no true value was found.

    pred : `collections.Callable` or `None`, optional
        If `None` find the last true value. If not `None` find the last value
        for which ``pred(value)`` is true.
        Default is ``None``.

    Examples
    --------
    >>> from nddata.utils.itertools_recipes import last_true
    >>> last_true([0, '', tuple(), 10])
    10

    >>> last_true([0,2,5,8,10], pred=lambda x: x%2)  # Last odd number
    5

    >>> last_true([0,0,0,0])
    False

    >>> last_true([0,0,0,0], default=100)  # default value if no true value.
    100
    """
    return next(tail(filter(pred, iterable), 1), default)
