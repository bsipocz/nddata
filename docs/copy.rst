When and how to copy
====================

This is just a cheatsheet including how types should be copied.

Singletons
----------

No need to copy the following types:

- `int`
- `float`
- `complex`
- `str`
- `bool`
- `None`

Mutables
--------

- `list` : ``alist[:]``
- `dict` : ``adict.copy()``
- `set` : ``aset.copy()``

Numpy
-----

- `numpy.ndarray` : ``ndarray.copy()``


Astropy
-------

- `astropy.wcs.WCS` : ``wcs.deepcopy()`` (``WCS.copy()`` is practically useless
  see also https://github.com/astropy/astropy/issues/4989)
- `astropy.io.fits.Header` : ``Header.copy()`` (``copy.copy(Header)`` does NOT
  really copy it! See also https://github.com/astropy/astropy/issues/4990)


Internals
---------

- `~nddata.nddata.NDDataBase` : ``NDDataBase.copy()``
- `~nddata.nddata.meta.NDUncertainty` : ``NDUncertainty.copy()``
