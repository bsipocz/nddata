# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from astropy.extern import six
from astropy.units import Quantity

import numpy as np

if six.PY2:  # pragma: no cover
    from future_builtins import zip


__all__ = ['wcs_compare', 'pix2world']


def wcs_compare(wcs1, wcs2, *args, **kwargs):
    """
    Compares two `~astropy.wcs.WCS` instances for equality by wrapping
    :meth:`astropy.wcs.Wcsprm.compare`.

    Parameters
    ----------
    wcs1, wcs2 : `~astropy.wcs.WCS`
        The two WCS instances for the comparison.

    args, kwargs :
        Additional parameters for :meth:`astropy.wcs.Wcsprm.compare`:

        - **cmp**: `int`
            The comparison flags (and associated values):

            - WCSCOMPARE_ANCILLARY: 1
            - WCSCOMPARE_TILING: 2
            - WCSCOMPARE_CRPIX: 4

            These can be combined using ``|``.
            Default is ``0``.

        - **tolerance**: number
            Allowed absolute difference.
            Default is ``0``.

    Returns
    -------
    equal : `bool`
        ``True`` if the WCS are equal enough or ``False`` if not.

    Notes
    -----
    This function can be used with
    :meth:`~nddata.nddata.mixins.NDArithmeticMixin.add` (and similar ones) as
    ``compare_wcs`` parameter.
    """
    return wcs1.wcs.compare(wcs2.wcs, *args, **kwargs)


def pix2world(wcs, data, mode='all'):
    """Get the world coordinates of all grid points of some data array.

    Parameters
    ----------
    wcs : `~astropy.wcs.WCS`
        The world coordinate system for the data.

    data : `numpy.ndarray`
        The data for which to calculate the coordinates. This is only used to
        determine the size and shape, the actual values are unaffected.

    mode : `str` {"all" | "wcs"}, optional
        Which transformations are applied:

        - ``"wcs"`` : Only the base world coordinate system transformations.
        - ``"all"`` : base world coordinate system transformations and
          distortions, if present.

        Default is ``"all"``.

    Returns
    -------
    coordinates : `list` of Quantities
        The coordinates for the ``data``. The first element is the coordinate
        along the first axis, the second element along the second axis and so
        on.

    Raises
    ------
    ValueError
        If the ``mode`` is not ``"all"`` or ``"wcs"``.

    See also
    --------
    astropy.wcs.WCS.all_pix2world : Used for ``mode="all"``.
    astropy.wcs.WCS.wcs_pix2world : Used for ``mode="wcs"``.
    astropy.wcs.Wcsprm.p2s : Low-level method used by the WCS methods.
    astropy.wcs.utils.pixel_to_skycoord : Similar but converts to SkyCoord.
    astropy.coordinates.SkyCoord.from_pixel : Similar but converts to SkyCoord.

    Examples
    --------
    I'll use synthetic `~astropy.wcs.WCS` here but it should work with every
    (also non-celestial) WCS object::

        >>> from astropy.wcs import WCS
        >>> import numpy as np
        >>> from nddata.utils.wcs import pix2world

        >>> wcs = WCS(naxis=1)
        >>> wcs.wcs.crpix = [1]
        >>> wcs.wcs.crval = [0]
        >>> wcs.wcs.cunit = ['nm']
        >>> wcs.wcs.cdelt = [1.5]

        >>> data = np.ones(6)

        >>> pix2world(wcs, data)
        [<Quantity [ 0. , 1.5, 3. , 4.5, 6. , 7.5] nm>]

    This works also with multiple axis::

        >>> wcs = WCS(naxis=2)
        >>> wcs.wcs.crpix = [1, 1]
        >>> wcs.wcs.crval = [0, 2]
        >>> wcs.wcs.cunit = ['nm', 'deg']
        >>> wcs.wcs.cdelt = [1.5, 1.2]

        >>> data = np.ones((5, 5))

        >>> pix2world(wcs, data, mode='wcs')
        [<Quantity [[ 0. , 0. , 0. , 0. , 0. ],
                    [ 1.5, 1.5, 1.5, 1.5, 1.5],
                    [ 3. , 3. , 3. , 3. , 3. ],
                    [ 4.5, 4.5, 4.5, 4.5, 4.5],
                    [ 6. , 6. , 6. , 6. , 6. ]] nm>,
         <Quantity [[ 2. , 3.2, 4.4, 5.6, 6.8],
                    [ 2. , 3.2, 4.4, 5.6, 6.8],
                    [ 2. , 3.2, 4.4, 5.6, 6.8],
                    [ 2. , 3.2, 4.4, 5.6, 6.8],
                    [ 2. , 3.2, 4.4, 5.6, 6.8]] deg>]

    The WCS doesn't contain any distortions so the ``mode`` actually doesn't
    make any difference. But if the distortions should be **neglected** one
    should set ``mode="wcs"``.
    """
    # Create an open grid containing all pixel numbers. pix2world can handle
    # broadcasting so an open grid suffices.
    grid = np.ogrid[tuple(slice(shape) for shape in data.shape)]

    # We need to append the origin used for the conversion to the grid because
    # Python2 cannot handle parameters after parameter unpacking:
    # (*grid, 0) and (*grid, origin=0) just won't work...
    grid.append(0)

    # Call the WCS pix2world function in the appropriate mode
    if mode == 'all':
        result = wcs.all_pix2world(*grid)
    elif mode == 'wcs':
        result = wcs.wcs_pix2world(*grid)
    else:
        raise ValueError("mode must be 'wcs' or 'all'.")

    # Convert the results to Quantities using the units from wcs.wcs.cunit.
    # These should be convertible to astropy.units.Quantity by default.
    result = [Quantity(data, unit=unit, copy=False)
              for data, unit in zip(result, wcs.wcs.cunit)]

    return result
