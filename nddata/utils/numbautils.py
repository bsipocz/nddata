# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from .sentinels import ParameterNotSpecified
from ..deps import OPT_DEPS

__all__ = ['convolve', 'interpolate']

if not OPT_DEPS['NUMBA']:  # pragma: no cover
    __doctest_skip__ = ['interpolate', 'convolve']


def interpolate(data, kernel, mask=ParameterNotSpecified):
    """Interpolation of the masked values of some data by convolution.

    .. note::
        Requires ``Numba`` to be installed.

    Parameters
    ----------
    data : `numpy.ndarray`, `numpy.ma.MaskedArray`, `~nddata.nddata.NDData`
        The data to interpolate.

    kernel : `numpy.ndarray`, `astropy.convolution.Kernel`
        The kernel (or footprint) for the interpolation. The sum of the
        ``kernel`` must not be 0 (or very close to it). Each axis of the kernel
        must be odd.

    mask : `numpy.ndarray`, optional
        Masked values in the ``data``. Elements where the mask is equivalent to
        1 (also ``True``) are interpreted as masked and are ignored during the
        interpolation. If not given use the mask of the data or if it has no
        mask either assume all the data is unmasked.

    Returns
    -------
    interpolated : `numpy.ndarray`
        The interpolated array.

    See also
    --------
    convolve

    Notes
    -----
    1. If the ``data`` parameter has a ``mask`` attribute then ``data.data``
       is interpreted as ``data`` and ``array.mask`` as ``mask`` parameter.
       This allows using `~numpy.ma.MaskedArray` objects as ``data`` parameter.

    2. If an explicit ``mask`` is given (even if it is ``None``) an implicit
       mask is ignored.

    3. No border handling is possible, if the kernel extends beyond the
       image these _outside_ values are treated as if they were masked.

    4. Since the kernel is internally normalized (in order to allow ignoring
       the masked values) the sum of the kernel must not be 0 because each
       masked element in the interpolated image would be ``Inf`` since it is
       divided by the sum of the kernel. Even a kernel sum close to zero should
       be avoided because otherwise floating value precision might become
       significant.

    5. Only implemented for 1D, 2D and 3D data.

    Examples
    --------
    Interpolate masked values for a 1D masked array:

        >>> from nddata.utils.numbautils import interpolate
        >>> import numpy as np

        >>> data = np.ma.array([1,1000,2], mask=[0, 1, 0])
        >>> interpolate(data, [1,1,1])
        array([ 1. ,  1.5,  2. ])

    Interpolate a 2D list with an astropy kernel::

        >>> from astropy.convolution.kernels import Box2DKernel
        >>> kernel = Box2DKernel(3)
        >>> data = [[1,1,1],[1,1000,1],[1,1,3]]
        >>> mask = [[0,0,0],[0,1,0],[0,0,0]]
        >>> interpolate(data, kernel, mask)
        array([[ 1.  ,  1.  ,  1.  ],
               [ 1.  ,  1.25,  1.  ],
               [ 1.  ,  1.  ,  3.  ]])

    Interpolate a 3D NDData instance::

        >>> from nddata.nddata import NDData
        >>> data = np.ones((3,3,3))
        >>> data[1,1,1] = 1000
        >>> mask = np.zeros((3,3,3))
        >>> mask[1,1,1] = 1
        >>> data = NDData(data, mask=mask)
        >>> kernel = np.ones((3,3,3))
        >>> interpolate(data, kernel)
        array([[[ 1.,  1.,  1.],
                [ 1.,  1.,  1.],
                [ 1.,  1.,  1.]],
        <BLANKLINE>
               [[ 1.,  1.,  1.],
                [ 1.,  1.,  1.],
                [ 1.,  1.,  1.]],
        <BLANKLINE>
               [[ 1.,  1.,  1.],
                [ 1.,  1.,  1.],
                [ 1.,  1.,  1.]]])

    If no mask is given the interpolation just returns the input::

        >>> interpolate([1,2,3], [1,1,1])
        array([1, 2, 3])

    Given an implicit and explicit mask the explicit mask is always used::

        >>> interpolate(np.ma.array([1,2,3], mask=[1,1,1]), [1,1,1], \
mask=[0,1,0])
        array([ 1.,  2.,  3.])
    """
    return _process(data, kernel, mask, 'interpolation')


def convolve(data, kernel, mask=ParameterNotSpecified, rescale_kernel=True):
    """Convolution of some data by ignoring masked values.

    .. note::
        Requires ``Numba`` to be installed.

    Parameters
    ----------
    data : `numpy.ndarray`, `numpy.ma.MaskedArray`, `~nddata.nddata.NDData`
        The data to convolve.

    kernel : `numpy.ndarray`, `astropy.convolution.Kernel`
        The kernel (or footprint) for the convolution. The sum of the
        ``kernel`` must not be 0 (or very close to it). Each axis of the kernel
        must be odd.

    mask : `numpy.ndarray`, optional
        Masked values in the ``data``. Elements where the mask is equivalent to
        1 (also ``True``) are interpreted as masked and are ignored during the
        convolution. If not given use the mask of the data or if it has no mask
        either assume all the data is unmasked.

    rescale_kernel : `bool`, optional
        In order to allow for ignoring masked values the kernel must be
        normalized (divided by it's sum). If this is ``True`` the result is
        multiplied with the sum again, if ``False`` the result of the
        convolution will be as if each kernel element was divided by the sum
        of the kernel.
        Default is ``True``.

    Returns
    -------
    convolved : `numpy.ndarray`
        The convolved array.

    See also
    --------
    interpolate : Another numba-based utility to only interpolate by \
        convolution.
    numpy.convolve : Fast one-dimensional convolution without masks.
    scipy.ndimage.convolve : Fast n-dimensional convolution without masks.
    astropy.convolution.convolve : Convolution that excludes ``NaN`` from the \
        data, without masks.

    Notes
    -----
    1. If the ``data`` parameter has a ``mask`` attribute then ``data.data``
       is interpreted as ``data`` and ``array.mask`` as ``mask`` parameter.
       This allows using `~numpy.ma.MaskedArray` objects as ``data`` parameter.

    2. If an explicit ``mask`` is given (even if it is ``None``) an implicit
       mask is ignored.

    3. No border handling is possible, if the kernel extends beyond the
       image these _outside_ values are treated as if they were masked.

    4. Since the kernel is internally normalized (in order to allow ignoring
       the masked values) the sum of the kernel must not be 0 because each
       element in the convolved image would be ``Inf`` since it is divided by
       the sum of the kernel. Even a kernel sum close to zero should be avoided
       because otherwise floating value precision might become significant.

    5. Only implemented for 1D, 2D and 3D data.

    Examples
    --------
    Convolution of a 1D masked array::

        >>> from nddata.utils.numbautils import convolve
        >>> import numpy as np

        >>> data = np.ma.array([1,1000,2,1], mask=[0, 1, 0, 0])
        >>> convolve(data, [1,1,1])
        array([ 1. ,  1.5,  1.5,  1.5])

    Convolution of a 2D list given an explicit mask and an astropy Kernel::

        >>> from astropy.convolution.kernels import Box2DKernel
        >>> data = [[1,1,1],[1,1000,1],[1,1,3]]
        >>> kernel = Box2DKernel(3)
        >>> mask = [[0,0,0],[0,1,0],[0,0,0]]
        >>> convolve(data, kernel, mask)
        array([[ 1.        ,  1.        ,  1.        ],
               [ 1.        ,  1.25      ,  1.4       ],
               [ 1.        ,  1.4       ,  1.66666667]])

    Convolution of a 3D NDData instance::

        >>> from nddata.nddata import NDData
        >>> data = np.ones((3,3,3))
        >>> data[1,1,1] = 1000
        >>> data[2,2,2] = 5
        >>> mask = np.zeros((3,3,3))
        >>> mask[1,1,1] = 1
        >>> data = NDData(data, mask=mask)
        >>> kernel = np.ones((3,3,3))
        >>> convolve(data, kernel)
        array([[[ 1.        ,  1.        ,  1.        ],
                [ 1.        ,  1.        ,  1.        ],
                [ 1.        ,  1.        ,  1.        ]],
        <BLANKLINE>
               [[ 1.        ,  1.        ,  1.        ],
                [ 1.        ,  1.15384615,  1.23529412],
                [ 1.        ,  1.23529412,  1.36363636]],
        <BLANKLINE>
               [[ 1.        ,  1.        ,  1.        ],
                [ 1.        ,  1.23529412,  1.36363636],
                [ 1.        ,  1.36363636,  1.57142857]]])

    If neither an explicit nor implicit mask is given it assumes all values are
    unmasked::

        >>> convolve([1,2,3], [1,1,1])
        array([ 1.5,  2. ,  2.5])

    An implicit mask can be ignored by setting the mask to ``None``::

        >>> convolve(np.ma.array([1,2,3], mask=[1,1,1]), [1,1,1], mask=None)
        array([ 1.5,  2. ,  2.5])

    Given an implicit and explicit mask the explicit mask is always used::

        >>> convolve(np.ma.array([1,2,3], mask=[1,1,1]), [1,1,1], mask=[0,1,0])
        array([ 1.,  2.,  3.])
    """
    return _process(data, kernel, mask, 'convolution')


def _process(data, kernel, mask, mode):
    """Convolution and Interpolation processing is much the same before the
    actual heavy lifting is performed. This function combines the processing
    and then invokes the appropriate numba-function.
    """
    if not OPT_DEPS['NUMBA']:  # pragma: no cover
        raise ImportError('{0} requires numba to be installed.'.format(mode))

    if mode not in ['convolution', 'interpolation']:
        raise ValueError('unknown type of mode {0}.'.format(mode))

    # Try to just extract a mask from the data. If it fails with an
    # AttributeError and no mask was specified create an empty boolean array.
    try:
        mask2 = data.mask
    except AttributeError:
        mask2 = None
    else:
        # In case we sucessfully used the mask of the data we either have a
        # numpy.ma.MaskedArray or a NDData instance so we set the data to
        # the data saved as the data attribute.
        data = data.data

    data = np.asarray(data)

    # Only in case no explicit mask was given use the one extracted from the
    # data.
    if mask is ParameterNotSpecified:
        mask = mask2

    # In case we have no mask (None) create an empyt one for convolution. In
    # case of interpolation we can exit early because there is nothing to
    # interpolate.
    if mask is None:
        if mode == 'convolution':
            mask = np.zeros(data.shape, dtype=bool)
        else:
            return data
    # Check if the shape is the same. There might be cases where the
    # array contained a mask attribute but the mask has a different shape
    # than the data!
    else:
        mask = np.asarray(mask)
        if data.shape != mask.shape:
            raise ValueError('data and mask must have the same shape.')

    # It is possible that the kernel is an astropy kernel, in that case it has
    # an attribute "array" and we use that one:
    kernel = np.asarray(getattr(kernel, 'array', kernel))

    # Evaluate how many dimensions the array has, this is needed to find the
    # appropriate convolution or interpolation function.
    ndim = data.ndim

    # kernel must have the same number of dimensions
    if kernel.ndim != ndim:
        raise ValueError('data and kernel must have the same number of '
                         'dimensions.')

    # the kernel also needs to be odd in every dimension.
    if any(i % 2 == 0 for i in kernel.shape):
        raise ValueError('kernel must have an odd shape in each dimension.')

    # Use the dictionary containing the appropriate functions for the
    # chosen mode and the given dimensions. We already checked that the mode is
    # ok therefore any KeyError must happen because the dimension was not
    # supported.
    try:
        return MODE_DIM_FUNC_MAP[mode][ndim](data, kernel, mask)
    except KeyError:
        raise ValueError('data must not have more than 3 dimensions.')


if OPT_DEPS['NUMBA']:  # pragma: no cover
    from numba import njit

    @njit
    def _interpolate_mask_1d(image, kernel, mask):
        nx = image.shape[0]
        nkx = kernel.shape[0]
        wkx = nkx // 2

        result = np.zeros(image.shape, dtype=np.float64)

        for i in range(0, nx, 1):
            if mask[i] == 1:
                iimin = max(i - wkx, 0)
                iimax = min(i + wkx + 1, nx)
                num = 0.
                div = 0.
                for ii in range(iimin, iimax, 1):
                    iii = wkx + ii - i
                    if mask[ii] == 0:
                        num += kernel[iii] * image[ii]
                        div += kernel[iii]
                if div == 0.0:
                    result[i] = np.nan
                else:
                    result[i] = num / div
            else:
                result[i] = image[i]
        return result

    @njit
    def _interpolate_mask_2d(image, kernel, mask):
        nx = image.shape[0]
        ny = image.shape[1]
        nkx = kernel.shape[0]
        nky = kernel.shape[1]
        wkx = nkx // 2
        wky = nky // 2

        result = np.zeros(image.shape, dtype=np.float64)

        for i in range(0, nx, 1):
            for j in range(0, ny, 1):
                if mask[i, j] == 1:
                    iimin = max(i - wkx, 0)
                    iimax = min(i + wkx + 1, nx)
                    jjmin = max(j - wky, 0)
                    jjmax = min(j + wky + 1, ny)
                    num = 0.
                    div = 0.
                    for ii in range(iimin, iimax, 1):
                        iii = wkx + ii - i
                        for jj in range(jjmin, jjmax, 1):
                            if mask[ii, jj] == 0:
                                jjj = wky + jj - j
                                num += kernel[iii, jjj] * image[ii, jj]
                                div += kernel[iii, jjj]
                    if div == 0.0:
                        result[i, j] = np.nan
                    else:
                        result[i, j] = num / div
                else:
                    result[i, j] = image[i, j]
        return result

    @njit
    def _interpolate_mask_3d(image, kernel, mask):
        nx = image.shape[0]
        ny = image.shape[1]
        nz = image.shape[2]
        nkx = kernel.shape[0]
        nky = kernel.shape[1]
        nkz = kernel.shape[2]
        wkx = nkx // 2
        wky = nky // 2
        wkz = nkz // 2

        result = np.zeros(image.shape, dtype=np.float64)

        for i in range(0, nx, 1):
            for j in range(0, ny, 1):
                for k in range(0, nz, 1):
                    if mask[i, j, k] == 1:
                        iimin = max(i - wkx, 0)
                        iimax = min(i + wkx + 1, nx)
                        jjmin = max(j - wky, 0)
                        jjmax = min(j + wky + 1, ny)
                        kkmin = max(k - wkz, 0)
                        kkmax = min(k + wkz + 1, nz)
                        num = 0.
                        div = 0.
                        for ii in range(iimin, iimax, 1):
                            iii = wkx + ii - i
                            for jj in range(jjmin, jjmax, 1):
                                jjj = wky + jj - j
                                for kk in range(kkmin, kkmax, 1):
                                    if mask[ii, jj, kk] == 0:
                                        kkk = wkz + kk - k
                                        num += (kernel[iii, jjj, kkk] *
                                                image[ii, jj, kk])
                                        div += kernel[iii, jjj, kkk]
                        if div == 0.0:
                            result[i, j, k] = np.nan
                        else:
                            result[i, j, k] = num / div
                    else:
                        result[i, j, k] = image[i, j, k]
        return result

    @njit
    def _convolve_with_mask_1d(image, kernel, mask):
        nx = image.shape[0]
        nkx = kernel.shape[0]
        wkx = nkx // 2

        result = np.zeros(image.shape, dtype=np.float64)

        for i in range(0, nx, 1):
            iimin = max(i - wkx, 0)
            iimax = min(i + wkx + 1, nx)
            num = 0.
            div = 0.
            for ii in range(iimin, iimax, 1):
                if mask[ii] == 0:
                    iii = wkx + ii - i
                    num += kernel[iii] * image[ii]
                    div += kernel[iii]
            if div == 0.0:
                result[i] = np.nan
            else:
                result[i] = num / div
        return result

    @njit
    def _convolve_with_mask_2d(image, kernel, mask):
        nx = image.shape[0]
        ny = image.shape[1]
        nkx = kernel.shape[0]
        nky = kernel.shape[1]
        wkx = nkx // 2
        wky = nky // 2

        result = np.zeros(image.shape, dtype=np.float64)

        for i in range(0, nx, 1):
            iimin = max(i - wkx, 0)
            iimax = min(i + wkx + 1, nx)
            for j in range(0, ny, 1):
                jjmin = max(j - wky, 0)
                jjmax = min(j + wky + 1, ny)
                num = 0.
                div = 0.
                for ii in range(iimin, iimax, 1):
                    iii = wkx + ii - i
                    for jj in range(jjmin, jjmax, 1):
                        if mask[ii, jj] == 0:
                            jjj = wky + jj - j
                            num += kernel[iii, jjj] * image[ii, jj]
                            div += kernel[iii, jjj]
                if div == 0.0:
                    result[i, j] = np.nan
                else:
                    result[i, j] = num / div
        return result

    @njit
    def _convolve_with_mask_3d(image, kernel, mask):
        nx = image.shape[0]
        ny = image.shape[1]
        nz = image.shape[2]
        nkx = kernel.shape[0]
        nky = kernel.shape[1]
        nkz = kernel.shape[2]
        wkx = nkx // 2
        wky = nky // 2
        wkz = nkz // 2

        result = np.zeros(image.shape, dtype=np.float64)

        for i in range(0, nx, 1):
            iimin = max(i - wkx, 0)
            iimax = min(i + wkx + 1, nx)
            for j in range(0, ny, 1):
                jjmin = max(j - wky, 0)
                jjmax = min(j + wky + 1, ny)
                for k in range(0, nz, 1):
                    kkmin = max(k - wkz, 0)
                    kkmax = min(k + wkz + 1, nz)
                    num = 0.
                    div = 0.
                    for ii in range(iimin, iimax, 1):
                        iii = wkx + ii - i
                        for jj in range(jjmin, jjmax, 1):
                            jjj = wky + jj - j
                            for kk in range(kkmin, kkmax, 1):
                                if mask[ii, jj, kk] == 0:
                                    kkk = wkz + kk - k
                                    num += (kernel[iii, jjj, kkk] *
                                            image[ii, jj, kk])
                                    div += kernel[iii, jjj, kkk]
                    if div == 0.0:
                        result[i, j, k] = np.nan
                    else:
                        result[i, j, k] = num / div
        return result

    MODE_DIM_FUNC_MAP = {'convolution':   {1: _convolve_with_mask_1d,
                                           2: _convolve_with_mask_2d,
                                           3: _convolve_with_mask_3d},

                         'interpolation': {1: _interpolate_mask_1d,
                                           2: _interpolate_mask_2d,
                                           3: _interpolate_mask_3d},
                         }
