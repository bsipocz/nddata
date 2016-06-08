# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from itertools import product

import numpy as np

from astropy import log
import astropy.units as u
from astropy.units import Quantity

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

        .. note::
            ``NumPy``-like slicing and indexing is possible using the regular
            ``ndd[...]`` syntax where ``...`` is whatever you want to use as
            slice or index.

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
        nddata.utils.numpyutils.create_slices : Function to convert indices to\
            slices.
        nddata.nddata.mixins.NDSlicingMixin.slice_cutout : Allows specifying \
            WCS coordinates

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

    def slice_cutout(self, point, shape, origin='start'):
        """Like :meth:`slice` but also allows giving position or shape in the \
                coordinate frame of the saved wcs.

        .. note::
            This method **requires** that the ``wcs`` attribute is an
            `~astropy.wcs.WCS` object containing **accurate** information about
            the saved ``data``.

        Parameters
        ----------
        position, shape : `int`, `~astropy.units.Quantity` or `tuple` of those
            Additionally to the options in :meth`slice` these can also be
            based on Quantities. In that case the parameter is evaluated in the
            coordinate system provided as ``wcs``.

            .. warning::
                The ``shape`` is interpreted as the total extend of the result
                and not as radius!

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
            If the number of positions or shape does't match the number of axis
            set in the wcs.

        See also
        --------
        nddata.nddata.mixins.NDSlicingMixin.slice : Does the slicing after the\
            coordinates are converted to grid points.
        astropy.wcs.WCS.all_world2pix : Convert coordinates to grid points.

        Notes
        -----
        Even though position and shape can consist of either Quantities or
        normal indices each of those **must** contain either Quantities **or**
        indices. For example:
        ``position=(10, 5), shape=(100*u.degree, 20*u.degree)`` is allowed but
        ``position=(10, 5 * u.degree), shape=(100*u.degree, 20*u.degree)`` is
        **NOT**, because the position contains degrees and normal indices.

        Also if using Quantities the results are converted and rounded. This
        could yield resulting shapes that are off by one compared to normal
        slicing. But otherwise it would be ambiguous if the start and end point
        should be included or not. Rounding to the nearest integer seems more
        likely to be expected of a cutout.

        Quantities are converted to the ``cunit`` of the ``wcs.wcs`` attribute.
        If you want to enable equivalencies have a look at the context manager
        of `astropy.units.equivalencies` and if necessary make sure the
        ``cunit`` is correct!

        Examples
        --------
        Supposing you have a wcs where one axis is degrees and the other one
        is some length (spectrum)::

            >>> import numpy as np
            >>> from astropy.wcs import WCS

            >>> wcs = WCS(naxis=2)
            >>> wcs.wcs.crpix = (1, 1)
            >>> wcs.wcs.crval = (120, 300)
            >>> wcs.wcs.cunit = ["deg", "nm"]
            >>> wcs.wcs.cdelt = [1, 5]

        And some data with a shape of 10x50::

            >>> data = np.arange(500).reshape(10, 50)
            >>> from nddata.nddata import NDData
            >>> ndd = NDData(data, wcs=wcs)

        Slicing from 123 degrees spanning 3 degrees and from 450 nanometers
        and 100 nanometers in shape::

            >>> ndd_cutout1 = ndd.slice_cutout((123*u.degree, 450*u.nm),
            ...                                (3*u.degree, 100*u.nm))
            >>> ndd_cutout1.data.shape
            (4, 20)

        or with specifying a center the shape will be equally distributed in
        either direction::

            >>> ndd_cutout2 = ndd.slice_cutout((125*u.degree, 450*u.nm),
            ...                                (3*u.degree, 50*u.nm),
            ...                                origin="center")
            >>> ndd_cutout2.data.shape
            (3, 11)

        Also possible is to give the position or shape in grid-coordinates by
        omitting the unit::

            >>> ndd_cutout3 = ndd.slice_cutout((3, 20),
            ...                                (3*u.degree, 50*u.nm),
            ...                                origin="end")
            >>> ndd_cutout3.data.shape
            (4, 11)

            >>> ndd_cutout4 = ndd.slice_cutout((125*u.degree, 470*u.nm),
            ...                                (4, 10),
            ...                                origin="end")
            >>> ndd_cutout4.data.shape
            (4, 10)

        If the resulting cutout is partially or totally outside the valid
        range for the data the result will only include the valid points and
        partially trimmed (or return an empty result in case the area is
        totally outside)::

            >>> ndd_cutout5 = ndd.slice_cutout((100*u.degree, 450*u.nm),
            ...                                (3*u.degree, 100*u.nm))
            >>> ndd_cutout5.data.shape
            (0, 20)

            >>> ndd_cutout6 = ndd.slice_cutout((132*u.degree, 450*u.nm),
            ...                                (3*u.degree, 100*u.nm), 'end')
            >>> ndd_cutout6.data.shape
            (1, 21)
        """
        wcs = self.wcs
        # Generally the position is an naxis-length iterable containing
        # quantities only in case of naxis == 1 we might also have just a
        # Quantity. We need to catch that case.
        if wcs.naxis == 1:
            # If it has no length it will raise a TypeError. Scalars (numpy,
            #  quantity, python, ...) have no length.
            try:
                len(point)
            except TypeError:
                point = (point, )

            # same for the shape
            try:
                len(shape)
            except TypeError:
                shape = (shape, )

        if wcs.naxis != len(point) or wcs.naxis != len(shape):
            raise ValueError('Shape of input ({0} and {1}) doesn\' match the '
                             'number of wcs axis ({2}).'
                             ''.format(len(point), len(shape), wcs.naxis))

        # We allow mixed position/shape interpretations.
        # 1.) grid-related: unitless numbers
        # 2.) wcs-related: quantities with a unit
        # There are some cases where the interpretation is ambiguous. Pixel and
        # dimensionless. These could be grid-related if the wcs is about
        # something else but could be wcs-related if the image is binned. I
        # just assume that the person REALLY meant them to be wcs-related.

        # To test which case we have we can check if the first item in
        # position and shape has a unit.
        pos_is_grid = not hasattr(point[0], 'unit')
        shape_is_grid = not hasattr(shape[0], 'unit')

        # There are several valid cases:
        # 1.) Position grid-related and shape grid-related
        #     Just forward to the normal slice-method.
        # 2.) Position grid-related and shape wcs-related
        #     We convert the position to a wcs-related coordinate and then do
        #     4.)
        # 3.) Position wcs-related and shape grid-related
        #     We calculate the pixel value of the position and then do 1.)
        # 4.) Position wcs-related and shape wcs-related:
        #     We compute the wcs-coordinates at the corners and convert these
        #     to pixel and create a slice with these.
        #     This might be inacurrate for convex wcs or if the off-diagonal
        #     terms are dominant.
        # The case where for example position contains wcs and grid related
        # terms are excluded (same for shape) because these would require
        # solving some ugly equations.

        # Convert units to the unit of the wcs (if they were given as
        # Quantities).
        # TODO: This seems to be the bottleneck for this function. Not sure
        # why. converting 3 quantities takes approximatly 200 us but the total
        # function runs for 800 us. The lprun shows that 50% is used in this
        # "if" so either the "value" access or the zip makes another 200 us (or
        # the combination of both). The same is true for the next if as well.
        if not pos_is_grid:
            point = [p.to(unit).value for p, unit in zip(point, wcs.wcs.cunit)]

        if not shape_is_grid:
            shape = [s.to(unit).value for s, unit in zip(shape, wcs.wcs.cunit)]

        # Now we can actually begin to catch the different cases. Case 1 and 3
        # will both end up calling the normal slicing. The common distinction
        # is that the shape is given in grid-coordinates:
        if shape_is_grid:
            # In case of case 3 we need to convert the wcs-related coordinate
            # to the corresponding pixel
            if not pos_is_grid:
                # Convert the position to an array containing at least 2
                # dimensions and wrap this as list and append the origin. This
                # allows using unpacking it when calling the function.
                # Otherwise Python2 throws a SyntaxError because it doesn't
                # allow unnamed parameters after unpacking a sequence.
                point = [np.array(point, ndmin=2), 0]
                # We convert it to a pixel value, round it to the nearest
                # integer. Then we discard the outer array since the position
                # is just one point and finally convert these to real integer.
                point = list(wcs.all_world2pix(*point).round()[0].astype(int))

            # Now we can simply call the normal slicing.
            return self.slice(point, shape, origin)

        # Here ends the cases 1 and 3. The following only applies to 2 and 4.

        # The shape will be wcs-related but the position might still be
        # grid-related. In this case we need to convert the grid-coordinates to
        # a wcs-coordinate.
        if pos_is_grid:
            point = [np.array(point, ndmin=2), 0]
            point = wcs.all_pix2world(*point)[0]

        # We now need to find the edges. These depend on the "origin" value,
        # we convert each of these to the "start" origin so we can calculate
        # the edges in one go.
        if origin != 'start':
            if origin == 'center':
                point = [pos_i - 0.5 * shape_i
                         for pos_i, shape_i in zip(point, shape)]
            if origin == 'end':
                point = [pos_i - shape_i
                         for pos_i, shape_i in zip(point, shape)]

        # Now the positions are all at the start-edge. So we only need to
        # calculate the position of the other edges. These are simply
        # start-point + shape for each dimension
        edges = [[pos_i, pos_i + shape_i]
                 for pos_i, shape_i in zip(point, shape)]
        # To get all edges we use itertools.product, and cast it to a list.
        # This is now a list of tuples, but all_world2pix requires a
        # "npoints x ndimensions" array so simply cast it to a numpy array
        edges = np.array(list(product(*edges)))
        # Calculate the pixel coordinates of each edge-point.
        edges = wcs.all_world2pix(edges, 0)
        # The start-position should be the minimum in each dimension which can
        # be calculated using the minimum along axis=0. This is a float but we
        # need an integer for correct slicing so round and int-cast it.
        point = edges.min(axis=0).round().astype(int)
        # The shape is just the maximum (rounded and cast as well) - the
        # minimum and also incremented by 1 to take care of the fact that we
        # WANT to include the end point.
        shape = list(edges.max(axis=0).round().astype(int) - point + 1)
        # Theoretically this position and shape can now be used by the normal
        # slicing as well.
        point = list(point)
        return self.slice(point, shape, origin='start')

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
