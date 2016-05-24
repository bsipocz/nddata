.. _nddata_stats:

.. doctest-skip-all

Statistics and NDData
=====================

.. warning::
    This page is not documentation tested. So if you find any problems while
    trying the examples please let me know. Thank you!

Introduction
------------

The `~nddata.nddata.NDData` class already implements the
`~nddata.nddata.mixins.NDStatsMixin` so the method
:meth:`~nddata.nddata.mixins.NDStatsMixin.stats` can be used without the need
of subclassing yourself.

By default it will contain some statistical information on the ``data`` based
on NumPy functions, like :func:`numpy.mean`. It collects these and returns a
`~astropy.table.Table` containing these::

    >>> from nddata.nddata import NDData
    >>> import numpy as np
    >>> data = np.random.randint(-100, 101, (1000, 1000))
    >>> ndd = NDData(data)
    >>> ndd.stats()
    elements min  max   mean  median mode      std           var      masked invalid
    -------- ---- --- ------- ------ ---- ------------- ------------- ------ -------
     1000000 -100 100 0.01389    0.0   12 58.0224568858 3366.60550307      0       0


.. note::
    The output for the example and the following examples will differ if you
    run the example yourself!

More statistics
---------------

To also get some ``AstroPy`` or ``SciPy`` related statistics you can set the
appropriate parameter to ``True``. However these might slow the execution
down. Especially the scipy functions.

SciPy::

    >>> from nddata.nddata import NDData
    >>> import numpy as np
    >>> data = np.random.normal(20, 2, (2000, 2000))
    >>> ndd = NDData(data)
    >>> stats = ndd.stats(scipy=True)
    >>> stats['skew']
           skew
    -----------------
    0.000141236222347

    >>> stats['kurtosis']
        kurtosis
    ----------------
    0.00558804679434

AstroPy::

    >>> stats2 = ndd.stats(astropy=True)
    >>> stats2['mad']
         mad
    -------------
    1.99823085467

    >>> stats2['biweight_location']
    biweight_location
    -----------------
        19.9997678852

    >>> stats2['biweight_midvariance']
    biweight_midvariance
    --------------------
           2.01787811593

Statistics exluding elements
----------------------------

You might have noticed the ``masked`` and ``invalid`` columns in the first
example. The reason for these columns is that the statistics excludes masked
elements, provided that the ``mask`` is a `numpy.ndarray` containing booleans
where ``True`` indicates invalid values and ``False`` invalid ones. The
``True`` elements are discarded before the statistic functions are run. For
example::

    >>> data = np.random.randint(-100, 101, (1000, 1000))
    >>> mask = data < 0
    >>> ndd = NDData(data, mask=mask)
    >>> stats = ndd.stats()
    >>> stats.pprint(max_width=-1)
    elements min max      mean     median mode      std           var      masked invalid
    -------- --- --- ------------- ------ ---- ------------- ------------- ------ -------
      502769   0 100 49.9703800354   50.0   25 29.1383831488 849.045372526 497231       0

Here the ``elements`` indicate how many elements were used for the statistical
properties and the ``masked`` column shows how many were discarded. The
``invalid`` column is still empty. Invalid values are ``NaN`` or ``Inf``, for
example::

    >>> data = np.random.normal(10, 5, (1000, 1000))
    >>> ndd = NDData(data)
    >>> ndd.data[ndd.data < 0] = np.nan # Set all negative elements to NaN
    >>> stats = ndd.stats()
    >>> stats.pprint(max_width=-1)
    elements        min             max           mean         median    mode      std           var      masked invalid
    -------- ----------------- ------------- ------------- ------------- ---- ------------- ------------- ------ -------
      977165 7.37029104254e-05 32.3342666175 10.2817126633 10.1479642207 10.0 4.70811522905 22.1663490101      0   22835

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

    >>> data = np.random.normal(10, 5, (1000, 1000))
    >>> ndd = NDData(data)
    >>> ndd.stats()['mode']
    mode
    ----
    10.0
    >>> ndd.stats(decimals_mode=0)['mode']
    mode
    ----
    10.0

A positive value, for example 2 will round the data to the nearest second
digit::

    >>> ndd.stats(decimals_mode=2)['mode']  # round to 2 decimals
    mode
    ----
    9.75

and negative values will round it to the 10s, so a value of -2 will round it
to the nearest 100::

    >>> ndd.stats(decimals_mode=-2)['mode']  # round to 2 digits before decimals (nearest 100)
    mode
    ----
     0.0
    >>> ndd.stats(decimals_mode=-1)['mode']  # round to 1 digits before decimals (nearest 10)
    mode
    ----
    10.0

The reason for this approximation is two-fold. The alternative
:func:`scipy.stats.mode` is extremly slow and using this rounding can speed
this up by a factor of 10-100. The other reason is that data containing
floating point values is **very unlikely** to have one value more than once,
except in some rare circumstances. For example::

    >>> from scipy.stats import mode
    >>> import numpy as np
    >>> mode(np.random.random(10000))
    ModeResult(mode=array([  1.72841355e-05]), count=array([1]))

which just returned the smalles element found in the array and with a count of
1. With :func:`nddata.utils.stats.mode` you can analyze this bahaviour::

    >>> from nddata.utils.stats import mode
    >>> mode(data)
    (0.0, 5034)

So the most-common integer is ``0`` with 5034 counts. Taking into account more
decimal places::

    >>> mode(data, decimals=1)
    (0.40000000000000002, 1035)

    >>> mode(data, decimals=2)
    (0.62, 122)

    >>> mode(data, decimals=3)
    (0.36399999999999999, 21)

    >>> mode(data, decimals=5)
    (0.017919999999999998, 3)

    >>> mode(data, decimals=10)
    (0.058778748399999997, 2)

so with 10 decimal places the most common value has only 2 occurences, taking
full precision will almost always, even with big datasets, return the smallest
element with 1 count. Choosing the right amount of ``decimals_mode`` is
essential here.

Just a note about timings::

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
