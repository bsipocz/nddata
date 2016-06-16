Documentation
=============

This is a private spinoff of the `astropy.nddata` module.

It offers a lot of customization but it's mostly about one class:
`~nddata.nddata.NDData`.

In it's simplest form is just a wrapper for a `numpy.ndarray`-like ``data``
but also supports the attributes:

- ``mask``
- ``uncertainty``
- ``unit``
- ``wcs``
- ``flags``
- ``meta``


.. toctree::
    :maxdepth: 2

    nddata/index.rst
    utils/index.rst

Meta informations

.. toctree::
    :maxdepth: 1

    copy.rst
    changelog
