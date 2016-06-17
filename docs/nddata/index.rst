.. _astropy_nddata:

*****************************************
N-dimensional datasets (`nddata.nddata`)
*****************************************

Introduction
============

The ``nddata`` package provides a uniform interface to N-dimensional datasets
(for tabulated data please have a look at `astropy.table`) in astropy through:

+ `~nddata.nddata.NDDataBase`: a basic container for `numpy.ndarray`-like data.
+ `~nddata.nddata.NDData`: like NDDataBase but with additional functionality
  like an reading/writing, simple arithmetic operations and slicing.
+ `~nddata.nddata.StdDevUncertainty`, `~nddata.nddata.VarianceUncertainty`,
  `~nddata.nddata.RelativeUncertainty` and `~nddata.nddata.UnknownUncertainty`:
  classes that can store and propagate uncertainties for a NDData object.
+ :ref:`nddata_utils`: General utility functions for operations on these
  classes, including a decorator to facilitate writing functions for such
  classes.

Further reading
===============

.. toctree::
   :maxdepth: 1

   nddata.rst
   collection.rst
   mixins/index.rst
   subclassing.rst
   interface.rst
   utils.rst
   decorator.rst
   experimental.rst

Getting started
===============

NDDataBase
----------

The primary purpose of `~nddata.nddata.NDDataBase` is to act as a *container*
for data, metadata, and other related information like a mask.

An `~nddata.nddata.NDDataBase` object can be instantiated by passing it an
n-dimensional `numpy` array::

    >>> import numpy as np
    >>> from nddata.nddata import NDDataBase
    >>> array = np.zeros((12, 12, 12))  # a 3-dimensional array with all zeros
    >>> ndd1 = NDDataBase(array)

or something that can be converted to an `numpy.ndarray`::

    >>> ndd2 = NDDataBase([1, 2, 3, 4])
    >>> ndd2
    NDDataBase([1, 2, 3, 4])

and can be accessed again via the ``data`` attribute::

    >>> ndd2.data
    array([1, 2, 3, 4])

It also supports additional properties like a ``unit`` or ``mask`` for the
data, a ``wcs`` (world coordinate system) and ``uncertainty`` of the data and
additional ``meta`` attributes:

    >>> data = np.array([1,2,3,4])
    >>> mask = data > 2
    >>> unit = 'erg / s'
    >>> from nddata.nddata import StdDevUncertainty
    >>> uncertainty = StdDevUncertainty(np.sqrt(data)) # representing standard deviation
    >>> meta = {'object': 'fictional data.'}
    >>> flags = np.zeros(data.shape)
    >>> from astropy.coordinates import SkyCoord
    >>> wcs = SkyCoord('00h42m44.3s', '+41d16m09s')
    >>> ndd = NDDataBase(data, mask=mask, unit=unit, uncertainty=uncertainty,
    ...                  meta=meta, wcs=wcs)
    >>> ndd
    NDDataBase([1, 2, 3, 4])

the representation does not show additional attributes but these can be
accessed like ``data`` above::

    >>> ndd.uncertainty
    StdDevUncertainty([ 1.        ,  1.41421356,  1.73205081,  2.        ])
    >>> ndd.mask
    array([False, False,  True,  True], dtype=bool)


NDData
------

Building upon this pure container `~nddata.nddata.NDData` implements:

+ a ``read`` and ``write`` method to access astropys unified file io interface.
+ simple arithmetics like addition, subtraction, division and multiplication.
+ slicing.

Instances are created in the same way::

    >>> from nddata.nddata import NDData
    >>> ndd = NDData(ndd)
    >>> ndd
    NDData([1, 2, 3, 4])

But also support arithmetic (:ref:`nddata_arithmetic`) like addition::

    >>> import astropy.units as u
    >>> ndd2 = ndd.add([4, 3, 2, 1] * u.erg / u.s)
    >>> ndd2
    NDData([ 5.,  5.,  5.,  5.])

Because these operations have a wide range of options these are not available
using arithmetic operators like ``+``.

Slicing or indexing (:ref:`nddata_slicing`) is possible (issuing warnings if
some attribute cannot be sliced)::

    >>> ndd2[2:]  # discard the first two elements
    INFO: wcs cannot be sliced. [nddata.nddata.mixins.ndslicing]
    NDData([ 5.,  5.])
    >>> ndd2[1]   # get the second element
    INFO: wcs cannot be sliced. [nddata.nddata.mixins.ndslicing]
    NDData(5.0)

Creating from a file or writing (:ref:`nddata_io`) such an instance a file is
possible using :meth:`~nddata.nddata.mixins.NDIOMixin.read` and
:meth:`~nddata.nddata.mixins.NDIOMixin.write`. See these methods which formats
are supported. For example writing a ``.fits`` file:

    >>> # fits doesn't work with a single coordinate as wcs:
    >>> ndd2.wcs = None
    >>> ndd2.write('testfile.fits', format='simple_fits')

and reading this file::

    >>> ndd_fits = NDData.read('testfile.fits', format='simple_fits')
    >>> ndd_fits
    NDData([ 5.,  5.,  5.,  5.])


StdDevUncertainty
-----------------

`~nddata.nddata.StdDevUncertainty` implements uncertainty based on standard
deviation and can propagate these using the arithmetic methods of
`~nddata.nddata.NDData`::

    >>> from nddata.nddata import NDData, StdDevUncertainty
    >>> import numpy as np

    >>> uncertainty = StdDevUncertainty(np.arange(5))
    >>> ndd = NDData([5,5,5,5,5], uncertainty=uncertainty)

    >>> doubled_ndd = ndd.multiply(2)  # multiply by 2
    >>> doubled_ndd.uncertainty
    StdDevUncertainty([ 0.,  2.,  4.,  6.,  8.])

    >>> ndd2 = ndd.add(doubled_ndd)    # add the doubled to the original
    >>> ndd2.uncertainty
    StdDevUncertainty([ 0.        ,  2.23606798,  4.47213595,  6.70820393,
                        8.94427191])

    >>> # or taking into account that both of these uncertainties are correlated
    >>> ndd3 = ndd.add(doubled_ndd, uncertainty_correlation=1)
    >>> ndd3.uncertainty
    StdDevUncertainty([  0.,   3.,   6.,   9.,  12.])

.. note::
    The "amount" of correlation must be given, so ``1`` means correlated, ``-1``
    anti-correlated and ``0`` (default) uncorrelated. See also
    :ref:`nddata_arithmetic` for more information about correlation handling.

Reference/API
=============

.. automodapi:: nddata.nddata

.. automodapi:: nddata.nddata.mixins
    :no-inheritance-diagram:

.. automodapi:: nddata.nddata.meta
    :no-inheritance-diagram:

.. automodapi:: nddata.nddata.exceptions
    :no-inheritance-diagram:

.. automodapi:: nddata.nddata.utils
    :no-inheritance-diagram:
