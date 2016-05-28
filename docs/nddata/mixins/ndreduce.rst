.. _nddata_reduce:

Reduce functions for NDDataBase
===============================

Reduce functions are generally functions that collapse one dimension of the
data. For example :func:`numpy.mean` is such a function because it maps an
`numpy.ndarray` with several elements to one value.

For `~nddata.nddata.NDDataBase` there is the
`~nddata.nddata.mixins.NDStatsMixin` which contains several functions that
reduce the ``data`` to a scalar value. The
`~nddata.nddata.mixins.NDReduceMixin` on the other hand contains methods to
reduce only one dimension and not necessarily to a scalar. This resembles the
behaviour if you give an not ``None`` ``axis`` parameter to :func:`numpy.mean`.
The result will be another array but reduced by one dimension.


Mean and average
----------------

To calculate the ``mean`` just call
:meth:`~nddata.nddata.mixins.NDReduceMixin.reduce_mean`::

    >>> from nddata.nddata import NDData
    >>> import numpy as np

    >>> ndd = NDData([[1,1,3], [2,1,3], [5,2,1]])
    >>> mean0 = ndd.reduce_mean()
    >>> mean0
    NDData([ 2.66666667,  1.33333333,  2.33333333])

Also the error of the mean is calculated and saved as ``uncertainty``:

    >>> mean0.uncertainty
    VarianceUncertainty([ 0.96296296,  0.07407407,  0.2962963 ])

The ``axis`` parameter indicates along which axis (dimension) the result is
calculated. The default is ``0`` so the previous example calculated the mean
along the first dimension. You can change it in case you need to apply it to
a different dimension::

    >>> mean1 = ndd.reduce_mean(axis=1)
    >>> mean1
    NDData([ 1.66666667,  2.        ,  2.66666667])

    >>> mean1.uncertainty
    VarianceUncertainty([ 0.2962963 ,  0.22222222,  0.96296296])

If a boolean mask was present this will be taken into account during the
calculation::

    >>> ndd = NDData([[1,1,3], [2,1,3], [5,2,1]])
    >>> ndd.mask = np.array([[0,1,0], [1,1,1], [0,1,1]], dtype=bool)

    >>> mean0 = ndd.reduce_mean(axis=0)
    >>> mean0
    NDData([ 3.,  0.,  3.])
    >>> mean0.uncertainty
    VarianceUncertainty([ 2.,  0.,  0.])

    >>> mean1 = ndd.reduce_mean(axis=1)
    >>> mean1
    NDData([ 2.,  0.,  5.])
    >>> mean1.uncertainty
    VarianceUncertainty([ 0.5,  0. ,  0. ])

To apply also ``weights`` you need to call
:meth:`~nddata.nddata.mixins.NDReduceMixin.reduce_average`. The weights need to
have the same shape as the ``data`` or the same size as the axis along which
the average is computed::

    >>> ndd = NDData([[1,1,3], [2,1,3], [5,2,1]])

    >>> avg = ndd.reduce_average(axis=0, weights=[1,0,1])
    >>> avg
    NDData([ 3. ,  1.5,  2. ])
    >>> avg.uncertainty
    VarianceUncertainty([ 2.   ,  0.125,  0.5  ])

    >>> avg = ndd.reduce_average(axis=1, weights=[1,0,1])
    >>> avg
    NDData([ 2. ,  2.5,  3. ])
    >>> avg.uncertainty
    VarianceUncertainty([ 0.5  ,  0.125,  2.   ])

The averaging also works if a mask is set::

    >>> ndd = NDData([[1,1,3], [2,1,3], [5,2,1]])
    >>> ndd.mask = np.array([[0,1,0], [0,0,0], [0,1,1]], dtype=bool)

    >>> avg = ndd.reduce_average(axis=0, weights=[1,0,1])
    >>> avg
    NDData([ 3.,  0.,  3.])
    >>> avg.uncertainty
    VarianceUncertainty([ 2.,  0.,  0.])

    >>> avg = ndd.reduce_average(axis=1, weights=[1,0,1])
    >>> avg
    NDData([ 2. ,  2.5,  5. ])
    >>> avg.uncertainty
    VarianceUncertainty([ 0.5  ,  0.125,  0.   ])


Median
------

:meth:`~nddata.nddata.mixins.NDReduceMixin.reduce_median` will give you the
``median`` along an axis. This also respects the ``mask``, if set, but the
resulting uncertainty is the corrected median absolute deviation::

    corrected_mad = 1.4826[...] * median_absolute_deviation
    corrected_mad = corrected_mad / sqrt(number_of_valid_values)

and returned as `~nddata.nddata.StdDevUncertainty`.


    >>> ndd = NDData([[1,1,3], [2,1,3], [5,2,1]])
    >>> ndd.mask = np.array([[0,1,0], [0,0,0], [0,1,1]], dtype=bool)

    >>> median = ndd.reduce_median(axis=0)
    >>> median
    NDData([ 2.,  1.,  3.])
    >>> median.uncertainty
    StdDevUncertainty([ 0.85598079,  0.        ,  0.        ])

    >>> median = ndd.reduce_median(axis=1)
    >>> median
    NDData([ 2.,  2.,  5.])
    >>> median.uncertainty
    StdDevUncertainty([ 1.04835808,  0.85598079,  0.        ])

But the median absolute deviation is a bad indicator as standard deviation for
small arrays. This is clearly visible for the same NDData instance without
a mask::

    >>> ndd = NDData([[1,1,3], [2,1,3], [5,2,1]])

    >>> median = ndd.reduce_median(axis=0)
    >>> median
    NDData([ 2.,  1.,  3.])
    >>> median.uncertainty
    StdDevUncertainty([ 0.85598079,  0.        ,  0.        ])

    >>> median = ndd.reduce_median(axis=1)
    >>> median
    NDData([ 1.,  2.,  2.])
    >>> median.uncertainty
    StdDevUncertainty([ 0.        ,  0.85598079,  0.85598079])


Resulting uncertainty
---------------------

.. warning::
    The computation may change in the future. Currently the correction is done
    assuming a large population without degrees of freedom.

The resulting uncertainty is calculated using the error of the mean (or average
or median) which is basically the variance of the values divided by the number
of valid elements. This doesn't account for small samples where this should
(probably?) be divided by the number of valid element minus 1.

    >>> ndd = NDData([[1,1,3,4,1,2,1]])
    >>> ndd.reduce_mean(axis=1).uncertainty
    VarianceUncertainty([ 0.18075802])

And for comparison how it is internally calculated::

    >>> np.var(ndd.data) / ndd.data.size
    0.18075801749271139

This differs from the general accepted way of calculating it::

    >>> np.var(ndd.data) / (ndd.data.size - 1)
    0.21088435374149661

But this is not trivial to implement considering the ``mask`` and ``weights``
so it is currently **NOT** done.
