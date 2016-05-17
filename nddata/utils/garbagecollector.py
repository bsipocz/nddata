# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from collections import defaultdict
from gc import get_objects

__all__ = ['assert_memory_leak']


def assert_memory_leak(func, specific_objects=None):
    """Determines memory leaks based on :func:`gc.get_objects`.

    Parameters
    ----------
    func : callable
        A function that does something and should in theory after being
        finished not leak any objects.

    specific_objects : type or None, optional
        If None then it checks if any objects exist that didn't exist before
        the function was called. If given a class it checks only if objects
        of this class are present that were not before the function was called.

    Raises
    ------
    AssertionError
        If any objects or one specific object leaked.
    """
    before = defaultdict(int)
    after = defaultdict(int)

    for i in get_objects():
        before[type(i)] += 1
    func()
    for i in get_objects():
        after[type(i)] += 1

    if specific_objects is None:
        assert all(after[k] - before[k] == 0 for k in after)
    else:
        assert after[specific_objects] - before[specific_objects] == 0
