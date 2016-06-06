# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from astropy import log

from ...utils.numpyutils import create_slices


__all__ = ['NDSlicingMixin']


class NDSlicingMixin(object):
    """Mixin to provide slicing on objects using the `~.NDDataBase` interface.

    The ``data``, ``mask``, ``uncertainty``, ``flags`` and ``wcs`` will be
    sliced, if set and sliceable. The ``unit`` and ``meta`` will be untouched.

    .. warning::
        The sliced return will be, if possible, a reference and not a copy.

    Examples
    --------
    `~.NDData` implements this mixin::

        >>> from nddata.nddata import NDData
        >>> import numpy as np

        >>> ndd = NDData([1,2,3], mask=np.array([True, False, True]))
        >>> ndd[:2]
        NDData([1, 2])

        >>> ndd[~ndd.mask]
        NDData([2])
    """
    def __getitem__(self, item):
        # Abort slicing if the data is a single scalar.
        if self.data.shape == ():
            raise TypeError('scalars cannot be sliced.')

        # Let the other methods handle slicing.
        kwargs = self._slice(item)
        return self.__class__(**kwargs)

    def slice(self, point, shape, origin='start'):
        """Slice the `~nddata.nddata.NDDataBase` instance by choosing a \
                reference point and shape.

        This is a wrapper for :func:`~nddata.utils.numpyutils.create_slices`.

        Parameters
        ----------
        point : `int`, `tuple` of integers
            The position represents the starting/central/end point (inclusive)
            of the slice. The interpretation of the point is controlled by the
            ``origin`` parameter.

        shape : `int`, `tuple` of integers
            The shape represents the extend of the slice. The ``shape`` can
            also be a `numpy.ndarray` in which case it's shape is used.

            .. note::
                The ``point`` and ``shape`` should contain as many integer as
                the target array has dimensions. In case it is a flat (1D)
                array the parameters don't need to be tuples but can also be
                single integer. **But** both parameters must be the same type
                and contain the same number of elements.

        origin : `str` {"start" | "end" | "center"}, optional
            Defines the interpretation of the ``point`` parameter:

            - ``"start"``, first point included in the slice.
            - ``"end"``, last point included in the slice.
            - ``"center"``, central point of the slice. Odd shapes have as
              many elements before and after the center while even shapes have
              one more element before.

            Default is ``"start"``.

        Returns
        -------
        sliced nddatabase : `~nddata.nddata.NDDataBase`-like
            The sliced NDDataBase-like instance. When possible the items are
            references to the originals and not copies.

        Raises
        ------
        ValueError
            If the ``origin`` is a not allowed type or string.

        See also
        --------
        nddata.utils.numpyutils.create_slices

        Examples
        --------
        Besides the Python slicing syntax also this specific method can be used
        to slice an instance::

            >>> from nddata.nddata import NDData
            >>> import numpy as np
            >>> ndd = NDData(np.arange(20).reshape(4, 5))

        To slice a 2x3 shaped portion starting at 1, 1::

            >>> ndd.slice((1,1), (2,3), origin="start")
            NDData([[ 6,  7,  8],
                    [11, 12, 13]])

        This is equivalent to slicing a 2x3 portion ending at 2, 3::

            >>> ndd.slice((2,3), (2,3), origin="end")
            NDData([[ 6,  7,  8],
                    [11, 12, 13]])

        Or using the center at 2,2::

            >>> ndd.slice((2,2), (2,3), origin="center")
            NDData([[ 6,  7,  8],
                    [11, 12, 13]])

        Notes
        -----
        This method has more options of specifying an ``origin`` but doesn't
        allow for ``NumPy`` integer or boolean indexing. If you want to use
        those you can use the normal ``ndd[indices]`` syntax.
        """
        # Create the appropriate real slices and pass them to getitem.
        return self[create_slices(point, shape, origin)]

    def _slice(self, item):
        """
        Collects the sliced attributes and passes them back as `dict`.

        It passes uncertainty, mask and wcs to their appropriate ``_slice_*``
        method, while ``meta`` and ``unit`` are simply taken from the original.
        The data is assumed to be sliceable and is sliced directly.

        When possible the return should *not* be a copy of the data but a
        reference.

        Parameters
        ----------
        item : slice
            The slice passed to ``__getitem__``.

        Returns
        -------
        dict :
            Containing all the attributes after slicing - ready to
            use them to create ``self.__class__.__init__(**kwargs)`` in
            ``__getitem__``.
        """
        kwargs = {}
        kwargs['data'] = self.data[item]
        # Try to slice some attributes
        kwargs['uncertainty'] = self._slice_uncertainty(item)
        kwargs['mask'] = self._slice_mask(item)
        kwargs['wcs'] = self._slice_wcs(item)
        kwargs['flags'] = self._slice_flags(item)
        # Attributes which are copied and not intended to be sliced
        kwargs['unit'] = self.unit
        kwargs['meta'] = self.meta
        return kwargs

    def _slice_uncertainty(self, item):
        if self.uncertainty is None:
            return None
        try:
            return self.uncertainty[item]
        except TypeError:
            # Catching TypeError in case the object has no __getitem__ method.
            # But let IndexError raise.
            log.info("uncertainty cannot be sliced.")
        return self.uncertainty

    def _slice_mask(self, item):
        if self.mask is None:
            return None
        try:
            return self.mask[item]
        except TypeError:
            log.info("mask cannot be sliced.")
        return self.mask

    def _slice_wcs(self, item):
        if self.wcs is None:
            return None
        try:
            return self.wcs[item]
        except TypeError:
            log.info("wcs cannot be sliced.")
        return self.wcs

    def _slice_flags(self, item):
        if self.flags is None:
            return None
        try:
            return self.flags[item]
        except TypeError:
            log.info("flags cannot be sliced.")
        return self.flags
