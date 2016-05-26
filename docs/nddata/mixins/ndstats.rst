.. _nddata_stats:

Statistics and NDData
=====================

Introduction
------------

The `~nddata.nddata.NDData` class already implements the
`~nddata.nddata.mixins.NDStatsMixin` so the method
:meth:`~nddata.nddata.mixins.NDStatsMixin.stats` can be used without the need
of subclassing yourself.

By default it will contain some statistical information on the ``data`` based
on NumPy functions, like :func:`numpy.mean`. It collects these and returns a
`~astropy.table.Table` containing these.

But before I start showing how the results work I'll create two functions that
will always return the same random array so the results are reproducible::

    >>> import numpy as np
    >>> from astropy.utils.misc import NumpyRNGContext

    >>> def create_random_int_array():
    ...     with NumpyRNGContext(12):
    ...         data = np.random.randint(-100, 101, (1000, 1000))
    ...     return data

    >>> def create_random_normal_dist_array():
    ...     with NumpyRNGContext(54321):
    ...         data = np.random.normal(20, 2, (1000, 1000))
    ...     return data

    >>> def create_random_array_small():
    ...     with NumpyRNGContext(54321):
    ...         data = np.random.random(10000)
    ...     return data


Now let's come to the demonstration of the statistics::

    >>> from nddata.nddata import NDData
    >>> data = create_random_int_array()
    >>> ndd = NDData(data)
    >>> print(ndd.stats())
    elements min  max    mean   median mode      std          var     masked invalid
    -------- ---- --- --------- ------ ---- ------------- ----------- ------ -------
     1000000 -100 100 -0.006708    0.0   98 58.0462831799 3369.370991      0       0


More statistics
---------------

To also get some ``AstroPy`` or ``SciPy`` related statistics you can set the
appropriate parameter to ``True``. However these might slow the execution
down. Especially the scipy functions.

SciPy::

    >>> data = create_random_normal_dist_array()
    >>> ndd = NDData(data)
    >>> stats = ndd.stats(scipy=True)  # doctest: +SKIP
    >>> print(stats['skew'])  # doctest: +SKIP
           skew
    ------------------
    -0.000549681295095

    >>> print(stats['kurtosis'])  # doctest: +SKIP
         kurtosis
    -----------------
    -0.00499387426427

AstroPy::

    >>> stats2 = ndd.stats(astropy=True)
    >>> print(stats2['mad'])
         mad
    -------------
    2.00112451652

    >>> print(stats2['biweight_location'])
    biweight_location
    -----------------
        20.0010321296

    >>> print(stats2['biweight_midvariance'])
    biweight_midvariance
    --------------------
           2.01796334264

Statistics exluding elements
----------------------------

You might have noticed the ``masked`` and ``invalid`` columns in the first
example. The reason for these columns is that the statistics excludes masked
elements, provided that the ``mask`` is a `numpy.ndarray` containing booleans
where ``True`` indicates invalid values and ``False`` invalid ones. The
``True`` elements are discarded before the statistic functions are run. For
example::

    >>> data = create_random_int_array()
    >>> mask = data < 0
    >>> ndd = NDData(data, mask=mask)
    >>> stats = ndd.stats()
    >>> stats.pprint(max_width=-1)
    elements min max      mean     median mode      std           var      masked invalid
    -------- --- --- ------------- ------ ---- ------------- ------------- ------ -------
      502394   0 100 50.0313021254   50.0   98 29.1671158162 850.720645037 497606       0

Here the ``elements`` indicate how many elements were used for the statistical
properties and the ``masked`` column shows how many were discarded. The
``invalid`` column is still empty. Invalid values are ``NaN`` or ``Inf``, for
example::

    >>> data = create_random_normal_dist_array()
    >>> ndd = NDData(data)
    >>> ndd.data[ndd.data < 15] = np.nan # Set all negative elements to NaN
    >>> stats = ndd.stats()
    >>> stats.pprint(max_width=-1)
    elements      min          max           mean         median    mode      std           var      masked invalid
    -------- ------------- ------------ ------------- ------------- ---- ------------- ------------- ------ -------
      993752 15.0000298472 29.071808088 20.0367895889 20.0149876702 20.0 1.95418361454 3.81883359935      0    6248

Apparently the same could have been done with a mask but generally you don't
set elements to ``NaN`` but they will rather turn out to be ``NaN`` because of
some operation.

The mode
--------

The most common value, also called the **mode** is calculated by binning, so
the default return will always be an integer. If you want to take more (or
less) digits into account you can modify the **decimals_mode** parameter. For
example the value ``0`` is the default and will round the data to the nearest
even integer and then calculate the mode::

    >>> data = create_random_normal_dist_array()
    >>> ndd = NDData(data)
    >>> print(ndd.stats()['mode'])
    mode
    ----
    20.0
    >>> print(ndd.stats(decimals_mode=0)['mode'])
    mode
    ----
    20.0

A positive value, for example 2 will round the data to the nearest second
digit::

    >>> print(ndd.stats(decimals_mode=2)['mode'])  # round to 2 decimals
     mode
    -----
    19.88

and negative values will round it to the 10s, so a value of -2 will round it
to the nearest 100::

    >>> print(ndd.stats(decimals_mode=-2)['mode'])  # round to 2 digits before decimals (nearest 100)
    mode
    ----
     0.0
    >>> print(ndd.stats(decimals_mode=-1)['mode'])  # round to 1 digits before decimals (nearest 10)
    mode
    ----
    20.0

The reason for this approximation is two-fold. The alternative
:func:`scipy.stats.mode` is extremly slow and using this rounding can speed
this up by a factor of 10-100. The other reason is that data containing
floating point values is **very unlikely** to have one value more than once,
except in some rare circumstances. For example::

    >>> from scipy.stats import mode as scipy_mode  # doctest: +SKIP
    >>> data = create_random_array_small()
    >>> scipy_mode(data)  # doctest: +SKIP
    ModeResult(mode=array([ 0.00018641]), count=array([1]))

which just returned the smalles element found in the array and with a count of
1. With :func:`nddata.utils.stats.mode` you can analyze this bahaviour::

    >>> from nddata.utils.stats import mode
    >>> mode(data)
    (0.0, 5027)

So the most-common integer is ``0`` with 5034 counts. Taking into account more
decimal places::

    >>> mode(data, decimals=1)
    (0.40000000000000002, 1036)

    >>> mode(data, decimals=2)
    (0.28999999999999998, 124)

    >>> mode(data, decimals=3)
    (0.91800000000000004, 23)

    >>> mode(data, decimals=5)
    (0.021760000000000002, 3)

    >>> mode(data, decimals=10)
    (0.0001864096, 1)

so with 10 decimal places the most common value has only 1 occurence, taking
full precision will almost always, even with big datasets, return the smallest
element with 1 count. Choosing the right amount of ``decimals_mode`` is
essential here.

Just a note about timings (using ``SciPy 0.17.1``):

.. doctest-skip::

    >>> data = np.random.randint(0, 1000, 10000) # random integer
    >>> %timeit nddata_stats_mode(data, decimals=10)
    1000 loops, best of 3: 888 µs per loop
    >>> %timeit nddata_stats_mode(data, decimals=0)
    1000 loops, best of 3: 887 µs per loop
    >>> %timeit scipy_stats_mode(data)
    10 loops, best of 3: 128 ms per loop
    >>> data = np.random.randint(0, 1000, 50000) # more random integer
    >>> %timeit nddata_stats_mode(data, decimals=10)
    100 loops, best of 3: 3.78 ms per loop
    >>> %timeit nddata_stats_mode(data, decimals=0)
    100 loops, best of 3: 3.8 ms per loop
    >>> %timeit scipy_stats_mode(data)
    1 loop, best of 3: 341 ms per loop
    >>> data = np.random.random(10000)  # this time some floats
    >>> %timeit nddata_stats_mode(data, decimals=10)
    100 loops, best of 3: 3.31 ms per loop
    >>> %timeit nddata_stats_mode(data, decimals=0)
    100 loops, best of 3: 2.51 ms per loop
    >>> %timeit scipy_stats_mode(data)
    1 loop, best of 3: 1.16 s per loop

You can also see that for floating point inputs the number of decimals affects
the runtime. But not nearly as bad as for the scipy mode function.

.. note::
    ``Scipy 0.18`` will probably implement a comparably fast mode function so
    these timings will be inaccurate for future ``SciPy`` versions.
