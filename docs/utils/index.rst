*****************************
Utility functions and classes
*****************************


Console Utilities
=================

Console related utilities that alter or extend the standard output.

.. automodapi:: nddata.utils.console
    :no-inheritance-diagram:


Context Managers
================

Classes that can be used with ``with``.

.. automodapi:: nddata.utils.contextmanager
    :no-inheritance-diagram:


Copy Utilities
==============

Some classes do not implement appropriate ``__copy__`` methods and some should
not be copied. These functions help to find the right way to shallow copy an
arbitary variable.

.. automodapi:: nddata.utils.copyutils
    :no-inheritance-diagram:

Decorators
==========

Some common decorators used throughout nddata.

.. automodapi:: nddata.utils.decorators
    :no-inheritance-diagram:


Descriptors
===========

Some common descriptors used throughout nddata. For example a `property` is a
descriptor. An introduction to descriptors is given in the
`python documentation <https://docs.python.org/howto/descriptor.html>`_.

These descriptors are experimental and are not as efficient as normal
properties but they allow to reuse common attribute requirements.

.. automodapi:: nddata.utils.descriptors


Dictionary Utilities
====================

Some dictionary utility functions.

.. automodapi:: nddata.utils.dictutils


Spectral extraction Utilities
=============================

Some extraction utilities for spectra.

.. automodapi:: nddata.utils.extraction


Garbagecollector
================

Utility functions around pythons `gc` module.

.. automodapi:: nddata.utils.garbagecollector
    :no-inheritance-diagram:


Input Validation
================

Some functions that validate user input.

.. note::
    These may not be appropriate in most cases, only if try / except or
    the standard exceptions would be misleading these functions should be used.

.. automodapi:: nddata.utils.inputvalidation


Itertools recipes
=================

Python2 and 3 compatible `itertools` recipes - copied from the python module
documentation and only slightly modified to work across Python versions (and
with proper documentation).

.. automodapi:: nddata.utils.itertools_recipes


NumPy
=====

NumPy related utilities.

.. automodapi:: nddata.utils.numpyutils
    :no-inheritance-diagram:


Numba
=====

`Numba <http://numba.pydata.org/>`_ related utilities.

.. warning::
        These function require that `Numba <http://numba.pydata.org/>`_ is
        installed.

.. automodapi:: nddata.utils.numbautils


Sentinels
=========

Sentinels are classes where there is at most one instance, like `None` or
`True` and `False`. This module contains some custom sentinels and a
factory to create them.

.. automodapi:: nddata.utils.sentinels
    :no-inheritance-diagram:


Statistics
==========

Some very simple statistical tools.

.. automodapi:: nddata.utils.stats
    :no-inheritance-diagram:


WCS
===

Some convenience functions around `astropy.wcs.WCS`

.. automodapi:: nddata.utils.wcs
    :no-inheritance-diagram:
