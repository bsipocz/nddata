*****************************
Utility functions and classes
*****************************


Context Managers
================

Classes that can be used with ``with``.

.. automodapi:: nddata.utils.contextmanager
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

Some dictionary utility functions

.. automodapi:: nddata.utils.dictutils


Garbagecollector
================

Utility functions around pythons `gc` module.

.. automodapi:: nddata.utils.garbagecollector
    :no-inheritance-diagram:


NumPy
=====

NumPy related utilities.

.. automodapi:: nddata.utils.numpyutils
    :no-inheritance-diagram:


Sentinels
=========

Sentinels are classes where there is at most one instance, like `None` or
`True` and `False`. This module contains some custom sentinels and a
factory to create them.

.. automodapi:: nddata.utils.sentinels
    :no-inheritance-diagram:
