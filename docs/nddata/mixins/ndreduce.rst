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
