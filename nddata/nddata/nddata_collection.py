# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from ..utils.sentinels import ParameterNotSpecified
from ..utils.inputvalidation import as_unsigned_integer

__all__ = ['NDDataCollection']


class NDDataCollection(object):
    """Represents a collection of `~nddata.nddata.NDDataBase` instances.

    Parameters
    ----------
    ndds : `~nddata.nddata.NDDataBase`-like
        An arbitary number of nddata-like objects. There are no restrictions
        on the classes or values of the attributes while creating an instance
        but some methods do.
    """
    def __init__(self, *ndds):
        self._num = len(ndds)
        self._ndds = ndds

    @property
    def num(self):
        """The number of NDData objects in the collection.
        """
        return self._num

    @property
    def ndds(self):
        """The NDData objects saved.
        """
        return self._ndds

    def stack(self, axis=0, func=None, *args, **kwargs):
        """Stack the NDData objects into one instance.

        Uses an approach like :func:`numpy.stack`.

        .. note::
            Expects the following attributes to be `numpy.ndarray` or `None`:

            - ``data`` (must not be ``None``)
            - ``mask``
            - ``flags``
            - ``uncertainty.data``

            If not ``None`` they must _all_ have the same shape. The resulting
            ``dtype`` is determined by the first occurence of that attribute.

            Also the following attributes must be identical for all instances:

            - ``unit``
            - ``__class__``

            and these must be identical for all instances where the attribute
            is set:

            - ``uncertainty.__class__``
            - ``uncertainty.unit``

            Additionally one restriction is that the first instance **must**
            have ``data`` that is not `None` (to determine the final shapes and
            the shapes to compare with).

            The following attributes will be simply taken from the **first**
            instance:

            - ``meta``
            - ``wcs``

        Parameters
        ----------
        axis : positive `int`, optional
            Stack along this axis. Stacking will expand the "to be stacked"
            attributes with an empty axis before stacking them. This resembles
            the behaviour of :func:`numpy.stack` and is not like
            `numpy.concatenate`.
            Default is ``0``.

        func : `collections.Callable` or `None`, optional
            A function that is applied to the ``ndd`` before it is stacked.
            This can be useful if the ``ndds`` are strings and they should be
            lazy loaded inside the stacking. If ``None`` the ``ndds`` are used
            as they are.

        args, kwargs :
            additional parameter for ``func``. These are ignored if ``func`` is
            ``None``. The call to the function is:
            ``func(ndd, *args, **kwargs)``.

        Returns
        -------
        stacked : `~nddata.nddata.NDDataBase`-like
            The stacked ndds as instance of the same class as the inputs.

        Examples
        --------
        If the inputs are of shape ``(3, )`` and we stack along ``axis=0`` the
        resulting shape will be ``(n, 3)`` with ``n`` being the number of
        instances used::

            >>> from nddata.nddata import NDDataCollection, NDDataBase, NDData
            >>> import numpy as np
            >>> ndds = NDDataCollection(NDDataBase([1,2,3]),
            ...                         NDDataBase([3,2,1]))
            >>> ndds.stack()
            NDDataBase([[1, 2, 3],
                        [3, 2, 1]])

        Similar stacking them along ``axis=1`` creates a result with shape
        ``(3, n)``::

            >>> ndds = NDDataCollection(NDData([1,2,3]), NDData([3,2,1]))
            >>> ndds.stack(axis=1)
            NDData([[1, 3],
                    [2, 2],
                    [3, 1]])

        But be careful if the data contains different dtypes::

            >>> ndds = NDDataCollection(NDData([1,2,3]), NDData([3.5,2,1]))
            >>> ndds.stack(axis=1)
            NDData([[1, 3],
                    [2, 2],
                    [3, 1]])

        Here the ``dtype`` is determined by the first ``data`` attribute (which
        is `int`) and the following ``data`` are cast to this dtype. So simply
        reversing the instances solves this::

            >>> ndds = NDDataCollection(NDData([3.5,2,1]), NDData([1,2,3]))
            >>> ndds.stack(axis=1)
            NDData([[ 3.5,  1. ],
                    [ 2. ,  2. ],
                    [ 1. ,  3. ]])

        Not all values have to be set for ``mask``, ``flags`` or
        ``uncertainty``::

            >>> ndd1 = NDData(np.ones((3,3)), mask=np.ones((3,3), dtype=bool))
            >>> ndd2 = NDData(np.ones((3,3))*2)
            >>> ndds = NDDataCollection(ndd1, ndd2)
            >>> ndds.stack(axis=0).mask
            array([[[ True,  True,  True],
                    [ True,  True,  True],
                    [ True,  True,  True]],
            <BLANKLINE>
                   [[False, False, False],
                    [False, False, False],
                    [False, False, False]]], dtype=bool)
            >>> ndds = NDDataCollection(ndd2, ndd1)
            >>> ndds.stack(axis=0).mask
            array([[[False, False, False],
                    [False, False, False],
                    [False, False, False]],
            <BLANKLINE>
                   [[ True,  True,  True],
                    [ True,  True,  True],
                    [ True,  True,  True]]], dtype=bool)

        Using a function is also possible, for example to convert all elements
        to a specific subclass::

            >>> ndd1 = np.ones((3, 3))*5
            >>> ndd2 = np.ma.array(np.ones((3, 3))/2, mask=np.ones((3, 3)))
            >>> ndds = NDDataCollection(ndd1, ndd2)
            >>> ndds.stack(func=NDData)
            NDData([[[ 5. ,  5. ,  5. ],
                     [ 5. ,  5. ,  5. ],
                     [ 5. ,  5. ,  5. ]],
            <BLANKLINE>
                    [[ 0.5,  0.5,  0.5],
                     [ 0.5,  0.5,  0.5],
                     [ 0.5,  0.5,  0.5]]])

        Or even if the ``ndds`` are ``strings`` to load them from the disc:

            >>> def load(ndd, *args, **kwargs):
            ...     return NDData.read(ndd, *args, **kwargs)

        See also :meth:`~nddata.nddata.NDData.read`. Especially for big NDData
        objects this can save a lot of memory if not all of them have to be
        kept in memory.
        """
        # axis must be an unsigned integer.
        axis = as_unsigned_integer(axis)

        # Load the ndds saved...
        ndds = self.ndds

        # Setup the invariant parameters.
        cls = ParameterNotSpecified
        unit = ParameterNotSpecified
        u_unit = ParameterNotSpecified
        u_cls = ParameterNotSpecified
        shape = ParameterNotSpecified

        # Setup the parameters that should be stacked.
        data = ParameterNotSpecified
        mask = ParameterNotSpecified
        flags = ParameterNotSpecified
        uncertainty = ParameterNotSpecified

        # Setup the attributes that are simply taken from the first instance.
        meta = ParameterNotSpecified
        wcs = ParameterNotSpecified

        for idx, ndd in enumerate(ndds):
            if func is not None:
                ndd = func(ndd, *args, **kwargs)
            # Compare the invariants for the class of the instance, the unit
            # and the shape (the latter one requires that the data is a numpy
            # array)!
            cls = self._stack_invariant(ndd.__class__, cls, 'class')
            unit = self._stack_invariant(ndd.unit, unit, 'unit')
            shape = self._stack_invariant(ndd.data.shape, shape, 'shape')
            # Create the final shape _only_ during the first iteration.
            if not idx:
                finalshape = list(shape)
                finalshape.insert(axis, self.num)
                # The wcs and meta are also taken from the first instance.
                wcs = ndd.wcs
                meta = ndd.meta

            # The data cannot be None, because we already checked it's shape.
            data = self._stack_stacks(ndd.data, data, finalshape, axis, idx)

            # If there is an uncertainty check the invariants: class and unit
            # then compare it against the data shape and finally stack the
            # uncertainty data.
            if ndd.uncertainty is not None:
                u_cls = self._stack_invariant(ndd.uncertainty.__class__, u_cls,
                                              'uncertainty class')
                u_unit = self._stack_invariant(ndd.uncertainty.unit, u_unit,
                                               'uncertainty unit')
                shape = self._stack_invariant(ndd.uncertainty.data.shape,
                                              shape, 'uncertainty shape')
                uncertainty = self._stack_stacks(ndd.uncertainty.data,
                                                 uncertainty, finalshape,
                                                 axis, idx)

            # Compare the mask shape if any mask is set and then stack it.
            if ndd.mask is not None:
                shape = self._stack_invariant(ndd.mask.shape, shape,
                                              'mask shape')
                mask = self._stack_stacks(ndd.mask, mask, finalshape, axis,
                                          idx)

            # Same for the flags
            if ndd.flags is not None:
                shape = self._stack_invariant(ndd.flags.shape, shape,
                                              'flags shape')
                flags = self._stack_stacks(ndd.flags, flags, finalshape, axis,
                                           idx)

        # Data must be set at some point because the first instance MUST have
        # some data.
        # Mask and flags need not be set if all of them were None.
        mask = None if mask is ParameterNotSpecified else mask
        flags = None if flags is ParameterNotSpecified else flags
        # for the uncertainty we need to wrap it up in a new instance but only
        # if it was set at some point
        if uncertainty is ParameterNotSpecified:
            uncertainty = None
        else:
            # in this case the class and the unit are PROBABLY set...
            uncertainty = u_cls(uncertainty, unit=u_unit, copy=False)

        return cls(data, mask=mask, flags=flags, uncertainty=uncertainty,
                   meta=meta, wcs=wcs, unit=unit, copy=False)

    def _stack_invariant(self, ndd_prop, ref_prop, name):
        """Compare the reference to the current property and raise an Exception
        if they aren't the same.

        The function returns the current one if no reference was determined
        yet.

        Parameters
        ----------
        ndd_prop : any type
            current property.

        ref_prop : any type
            reference property (set to "ParameterNotSpecified" if no reference
            is determined yet).

        name : string
            The name of the property, just used for the exception message.

        Returns
        -------
        reference : any type
            The reference property value.
        """
        # If there is no parameter specified as reference just return the
        # current property.
        if ref_prop is ParameterNotSpecified:
            return ndd_prop
        # Otheriwse compare them and return the reference if they are the same
        # or raise an Exception if they aren't equal.
        elif ref_prop == ndd_prop:
            return ref_prop
        else:
            raise ValueError('{0} must be the same for all ndds.'.format(name))

    def _stack_stacks(self, ndd_prop, ref_prop, shape, axis, idx):
        """Insert the property in the right slice of the result.

        Parameters
        ----------
        ndd_prop : `numpy.ndarray`
            The property of the current ndd. Should not be ``None``.

        ref_prop : `numpy.ndarray` or ``ParameterNotSpecified``
            If ``ParameterNotSpecified`` then create a new return (with the
            dtype of the current property value). Otherwise an earlier version
            is used.

        shape : iterable of ints
            The shape for creating a new array if ``ref_prop`` is
            ``ParameterNotSpecified``.

        axis : positive `int`
            The axis along which to insert.

        idx : positive `int`
            The index along the axis in which to insert it.

        Returns
        -------
        ref_prop : `numpy.ndarray`
            The reference property, optionally (if ndd_prop wasn't None) with
            the inserted ndd_prop.
        """
        # In case the current property is set we need an array to insert it.
        # So if the reference property is not yet created create an array of
        # the appropriate shape with the currents value dtype.
        if ref_prop is ParameterNotSpecified:
            ref_prop = np.zeros(shape, dtype=ndd_prop.dtype)
        # We need the correct position to insert the current property so create
        # an array to index the whole reference array but replace the dimension
        # where the axis is set by the current index. This slices the
        # appropriate section where we need to insert the reference.
        slicer = [slice(None) for _ in shape]
        slicer[axis] = idx
        # Insert the reference and return the reference.
        ref_prop[tuple(slicer)] = ndd_prop
        return ref_prop
