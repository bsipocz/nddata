# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from .sentinels import ParameterNotSpecified
from ..deps import OPT_DEPS

__all__ = ['convolve', 'convolve_median', 'interpolate', 'interpolate_median']

if not OPT_DEPS['NUMBA']:  # pragma: no cover
    __doctest_skip__ = ['convolve', 'convolve_median',
                        'interpolate', 'interpolate_median']


def interpolate(data, kernel, mask=ParameterNotSpecified):
    """Interpolation of the masked values of some data by convolution.

    .. note::
        Requires ``Numba``.

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
    interpolate_median : Numba-based utility to interpolate masked values \
        using median convolution.
    convolve : Numba-based utility to convolve using weighted summation.
    convolve_median : Numba-based utility to convolve using the median.

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


def convolve(data, kernel, mask=ParameterNotSpecified, rescale=True,
             var=False):
    """Convolution of some data by ignoring masked values.

    .. note::
        Requires ``Numba``.

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

    var : `bool`, optional
        Also calculate the weighted variance. This can be quite slow!
        Default is ``False``.

    Returns
    -------
    convolved : `numpy.ndarray`
        The convolved array.

    variance : `numpy.ndarray`, optional
        The variance. Only returned if the parameter ``var`` was true.


    See also
    --------
    interpolate : Numba-based utility to interpolate masked values using \
        weighted convolution.
    interpolate_median : Numba-based utility to interpolate masked values \
        using median convolution.
    convolve_median : Numba-based utility to convolve using the median.

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

    .. warning::
        Weighted summation (and also the standard deviation) are calculated
        with naive formulas. These might not be appropriate in all cases!

    Examples
    --------
    Convolution of a 1D masked array::

        >>> from nddata.utils.numbautils import convolve
        >>> import numpy as np

        >>> data = np.ma.array([1,1000,2,1], mask=[0, 1, 0, 0])
        >>> convolve(data, [1,1,1])
        array([ 3. ,  4.5,  4.5,  4.5])
        >>> convolve(data, [1,1,1], rescale=False)
        array([ 1. ,  1.5,  1.5,  1.5])

    The ``rescale`` parameter determines if a **sum** convolution (in case it
    is not given or ``True``) or a **mean** convolution (if ``False``) is
    performed.

    Convolution of a 2D list given an explicit mask and an astropy Kernel::

        >>> from astropy.convolution.kernels import Box2DKernel
        >>> data = [[1,1,1],[1,1000,1],[1,1,3]]
        >>> kernel = Box2DKernel(3)
        >>> mask = [[0,0,0],[0,1,0],[0,0,0]]
        >>> convolve(data, kernel, mask, rescale=False)
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
        >>> convolve(data, kernel, rescale=False)
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
        array([ 4.5,  6. ,  7.5])

    An implicit mask can be ignored by setting the mask to ``None``::

        >>> convolve(np.ma.array([1,2,3], mask=[1,1,1]), [1,1,1], mask=None)
        array([ 4.5,  6. ,  7.5])

    Given an implicit and explicit mask the explicit mask is always used::

        >>> convolve(np.ma.array([1,2,3], mask=[1,1,1]), [1,1,1], mask=[0,1,0])
        array([ 3.,  6.,  9.])

    Rescaling is also possible (in that case it's the mean of the convolved
    elements rather than the rescaled sum)::

        >>> convolve([1,2,3], [1,1,1], rescale=False)
        array([ 1.5,  2. ,  2.5])

    It is also possible to calculate the variance (without degrees of freedom
    correction)::

        >>> convolve([0, 100, 40, 40], [1, 2, 1], mask=[0, 0, 1, 0],
        ...          rescale=False, var=True)
        (array([ 33.33333333,  66.66666667,  70.        ,  40.        ]),
         array([ 2222.22222222,  2222.22222222,   900.        ,     0.        \
]))

    .. warning::
        If ``var=True`` the function returns 2 results instead of one!
    """
    result = _process(data, kernel, mask, 'convolution')
    if var:
        variance = _process(data, kernel, mask, 'convolution', expected=result)
    else:
        variance = None

    if rescale:
        result *= np.sum(getattr(kernel, 'array', kernel))

    if variance is None:
        return result
    else:
        return result, variance


def interpolate_median(data, kernel, mask=ParameterNotSpecified):
    """Interpolation of masked values in the data based on median convolution.

    .. note::
        Requires ``Numba``.

    Parameters
    ----------
    data : `numpy.ndarray`, `numpy.ma.MaskedArray`, `~nddata.nddata.NDData`
        The data to interpolate.

    kernel : `numpy.ndarray`, `astropy.convolution.Kernel`
        The kernel for the interpolation. One difference from normal
        interpolation is that the actual values of the kernel do not matter,
        except when it is zero then it won't use this element for the median
        computation.
        Each axis of the kernel must be odd.

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
    interpolate : Numba-based utility to interpolate masked values using \
        weighted convolution.
    convolve : Numba-based utility to convolve using weighted summation.
    convolve_median : Numba-based utility to convolve using the median.

    Notes
    -----
    1. If the ``data`` parameter has a ``mask`` attribute then ``data.data``
       is interpreted as ``data`` and ``array.mask`` as ``mask`` parameter.
       This allows using `~numpy.ma.MaskedArray` objects as ``data`` parameter.

    2. If an explicit ``mask`` is given (even if it is ``None``) an implicit
       mask is ignored.

    3. No border handling is possible, if the kernel extends beyond the
       image these _outside_ values are treated as if they were masked.

    4. Only implemented for 1D, 2D and 3D data.

    Examples
    --------
    It works almost like `interpolate` but based on the ``median`` instead of
    the **sum** or **mean** of the elements within the kernel::

        >>> from nddata.utils.numbautils import interpolate_median
        >>> import numpy as np

        >>> data = np.ma.array([1,1000,2,1], mask=[0, 1, 0, 0])
        >>> interpolate_median(data, [1,1,1])
        array([ 1. ,  1.5,  2. ,  1. ])

    Support for two dimensional arrays and masks is also implemented::

        >>> data = np.arange(9).reshape(3, 3)
        >>> data[1, 1] = 100
        >>> mask = np.zeros((3, 3), dtype=bool)
        >>> mask[1, 1] = 1
        >>> interpolate_median(data, np.ones((3,3)), mask)
        array([[ 0.,  1.,  2.],
               [ 3.,  4.,  5.],
               [ 6.,  7.,  8.]])

    And also for three dimensional arrays::

        >>> data = np.arange(27).reshape(3, 3, 3)
        >>> data[0, 0, 0] = 10000
        >>> mask = np.zeros((3, 3, 3))
        >>> mask[0, 0, 0] = 1
        >>> interpolate_median(data, np.ones((3, 3, 3)), mask)
        array([[[  9.,   1.,   2.],
                [  3.,   4.,   5.],
                [  6.,   7.,   8.]],
        <BLANKLINE>
               [[  9.,  10.,  11.],
                [ 12.,  13.,  14.],
                [ 15.,  16.,  17.]],
        <BLANKLINE>
               [[ 18.,  19.,  20.],
                [ 21.,  22.,  23.],
                [ 24.,  25.,  26.]]])

    Kernel elements of zero can also be used to use only specified elements for
    the interpolation::

        >>> data = np.ma.array([1,1000,2,1], mask=[0, 1, 0, 0])
        >>> interpolate_median(data, [1,0,0])
        array([ 1.,  1.,  2.,  1.])

    This kernel uses only the left element for the interpolation.

    .. note::
        The median calculation is done by a jitted version based on
        Insertionsort which becomes inefficient for large kernels (more than
        hundred or several hundred elements).
    """
    return _process(data, kernel, mask, 'interpolation', True)


def convolve_median(data, kernel, mask=ParameterNotSpecified, mad=False):
    """Median based convolution of some data by ignoring masked values.

    .. note::
        Requires ``Numba``.

    Parameters
    ----------
    data : `numpy.ndarray`, `numpy.ma.MaskedArray`, `~nddata.nddata.NDData`
        The data to convolve.

    kernel : `numpy.ndarray`, `astropy.convolution.Kernel`
        The kernel for the convolution. One difference from normal convolution
        is that the actual values of the kernel do not matter, except when it
        is zero then it won't use the element for the median computation.
        Each axis of the kernel must be odd.

    mask : `numpy.ndarray`, optional
        Masked values in the ``data``. Elements where the mask is equivalent to
        1 (also ``True``) are interpreted as masked and are ignored during the
        convolution. If not given use the mask of the data or if it has no mask
        either assume all the data is unmasked.

    mad : `bool` or ``"robust"``, optional
        Also calculate the median absolute deviation. If ``"robust"`` is chosen
        multiply the result with approximatly 1.4826.
        Default is ``False``.

    Returns
    -------
    convolved : `numpy.ndarray`
        The convolved array.

    mad : `numpy.ndarray`, optional
        The median absolute deviation of the convolved array. Only returned if
        ``mad`` was True.

    See also
    --------
    interpolate : Numba-based utility to interpolate masked values using \
        weighted convolution.
    interpolate_median : Numba-based utility to interpolate masked values \
        using median convolution.
    convolve : Numba-based utility to convolve using weighted summation.

    scipy.ndimage.median_filter : Fast n-dimensional convolution \
        without masks.

    Notes
    -----
    1. If the ``data`` parameter has a ``mask`` attribute then ``data.data``
       is interpreted as ``data`` and ``array.mask`` as ``mask`` parameter.
       This allows using `~numpy.ma.MaskedArray` objects as ``data`` parameter.

    2. If an explicit ``mask`` is given (even if it is ``None``) an implicit
       mask is ignored.

    3. No border handling is possible, if the kernel extends beyond the
       image these _outside_ values are treated as if they were masked.

    4. Only implemented for 1D, 2D and 3D data.

    Examples
    --------
    It works almost like `convolve` but based on the ``median`` instead of the
    **sum** or **mean** of the elements within the kernel::

        >>> from nddata.utils.numbautils import convolve_median
        >>> import numpy as np

        >>> data = np.ma.array([1,1000,2,1], mask=[0, 1, 0, 0])
        >>> convolve_median(data, [1,1,1])
        array([ 1. ,  1.5,  1.5,  1.5])

    Support for two dimensional arrays and masks is also implemented::

        >>> data = np.arange(9).reshape(3, 3)
        >>> data[1, 1] = 100
        >>> mask = np.zeros((3, 3), dtype=bool)
        >>> mask[1, 1] = 1
        >>> convolve_median(data, np.ones((3,3)), mask)
        array([[ 1.,  2.,  2.],
               [ 3.,  4.,  5.],
               [ 6.,  6.,  7.]])

    And also for three dimensional arrays::

        >>> data = np.arange(27).reshape(3, 3, 3)
        >>> data[0, 0, 0] = 10000
        >>> mask = np.zeros((3, 3, 3))
        >>> mask[0, 0, 0] = 1
        >>> convolve_median(data, np.ones((3, 3, 3)), mask)
        array([[[  9. ,   9. ,   7.5],
                [  9. ,   9. ,   9. ],
                [  9.5,  10. ,  10.5]],
        <BLANKLINE>
               [[ 12. ,  12. ,  12. ],
                [ 13. ,  13.5,  13.5],
                [ 14. ,  14.5,  15. ]],
        <BLANKLINE>
               [[ 15.5,  16. ,  16.5],
                [ 17. ,  17.5,  18. ],
                [ 18.5,  19. ,  19.5]]])

    Explictly using kernel elements to zero excludes those elements for the
    convolution::

        >>> data = np.ma.array([1,1000,2,1], mask=[0, 1, 0, 0])
        >>> convolve_median(data, [1,0,0])
        array([ nan,   1.,  nan,   2.])

    Here only the left element is used for the convolution. For the first
    element the left one is outside the data and for the third element the
    convolution element is masked so both of them result in ``NaN``.

    .. note::
        The median calculation is done by a jitted version based on
        Insertionsort which becomes inefficient for large kernels (more than
        hundred or several hundred elements).

    The function can optionally determine the median absolute deviation of the
    convolved array::

        >>> convolve_median([0, 100, 40, 40], [1, 2, 1], mask=[0, 0, 0, 1],
        ...                 mad=True)
        (array([ 50.,  40.,  70.,  40.]), array([ 50.,  40.,  30.,   0.]))

    or even calculate the robust median absolute deviation::

        >>> convolve_median([0, 100, 40, 40], [1, 2, 1], mask=[0, 0, 0, 1],
        ...                 mad='robust')
        (array([ 50.,  40.,  70.,  40.]),
         array([ 74.13011093,  59.30408874,  44.47806656,   0.        ]))

    which is just the normal deviation multiplied by the constant factor of
    approximatly ``1.4826``.

    .. warning::
        If ``mad=True`` the function returns 2 results instead of one!
    """
    result = _process(data, kernel, mask, 'convolution', True)
    if mad:
        mad_result = _process(data, kernel, mask, 'convolution', True,
                              expected=result)
        if mad == 'robust':
            mad_result *= 1.482602218505602
        return result, mad_result
    else:
        return result


def _process(data, kernel, mask, mode, median=False, expected=None):
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

    # Use the dictionary containing the appropriate functions.
    # Median convolution and interpolation has it's own mapping:
    if median:
        if expected is None:
            to_func = MODE_DIM_FUNC_MEDIAN_MAP
        else:
            to_func = MODE_DIM_FUNC_MEDIAN_STD_MAP
    else:
        if expected is None:
            to_func = MODE_DIM_FUNC_MAP
        else:
            to_func = MODE_DIM_FUNC_MAP_STD

    # No need to prepare the expected argument since it was calculated during
    # another convolution so it must be a numpy array with right dimensions and
    # so on.

    # We already checked that the mode is ok therefore any KeyError must happen
    # because the dimension was not supported.
    try:
        if expected is None:
            return to_func[mode][ndim](data, kernel, mask)
        else:
            # in case of interpolation just use the convolution error function.
            # this shouldn't be possible to reach without directly invoking
            # this private function but if I later want to add it it's
            # already here :-)
            return to_func['convolution'][ndim](data, kernel, mask, expected)
    except KeyError:
        raise ValueError('data must not have more than 3 dimensions.')


if OPT_DEPS['NUMBA']:  # pragma: no cover
    from numba import njit

    @njit
    def _interpolate_mask_1d(image, kernel, mask):
        """Determine the weighted average for each masked value."""
        nx = image.shape[0]
        nkx = kernel.shape[0]
        wkx = nkx // 2

        result = np.zeros(image.shape, dtype=np.float64)

        for i in range(0, nx, 1):
            if mask[i]:
                iimin = max(i - wkx, 0)
                iimax = min(i + wkx + 1, nx)
                num = 0.
                div = 0.
                for ii in range(iimin, iimax, 1):
                    iii = wkx + ii - i
                    if not mask[ii]:
                        num += kernel[iii] * image[ii]
                        div += kernel[iii]
                if div:
                    result[i] = num / div
                else:
                    result[i] = np.nan
            else:
                result[i] = image[i]
        return result

    @njit
    def _interpolate_mask_2d(image, kernel, mask):
        """Determine the weighted average for each masked value."""
        nx = image.shape[0]
        ny = image.shape[1]
        nkx = kernel.shape[0]
        nky = kernel.shape[1]
        wkx = nkx // 2
        wky = nky // 2

        result = np.zeros(image.shape, dtype=np.float64)

        for i in range(0, nx, 1):
            for j in range(0, ny, 1):
                if mask[i, j]:
                    iimin = max(i - wkx, 0)
                    iimax = min(i + wkx + 1, nx)
                    jjmin = max(j - wky, 0)
                    jjmax = min(j + wky + 1, ny)
                    num = 0.
                    div = 0.
                    for ii in range(iimin, iimax, 1):
                        iii = wkx + ii - i
                        for jj in range(jjmin, jjmax, 1):
                            if not mask[ii, jj]:
                                jjj = wky + jj - j
                                num += kernel[iii, jjj] * image[ii, jj]
                                div += kernel[iii, jjj]
                    if div:
                        result[i, j] = num / div
                    else:
                        result[i, j] = np.nan
                else:
                    result[i, j] = image[i, j]
        return result

    @njit
    def _interpolate_mask_3d(image, kernel, mask):
        """Determine the weighted average for each masked value."""
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
                    if mask[i, j, k]:
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
                                    if not mask[ii, jj, kk]:
                                        kkk = wkz + kk - k
                                        num += (kernel[iii, jjj, kkk] *
                                                image[ii, jj, kk])
                                        div += kernel[iii, jjj, kkk]
                        if div:
                            result[i, j, k] = num / div
                        else:
                            result[i, j, k] = np.nan
                    else:
                        result[i, j, k] = image[i, j, k]
        return result

    @njit
    def _convolve_with_mask_1d(image, kernel, mask):
        """Determine the weighted average for each value."""
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
                if not mask[ii]:
                    iii = wkx + ii - i
                    num += kernel[iii] * image[ii]
                    div += kernel[iii]
            if div:
                result[i] = num / div
            else:
                result[i] = np.nan
        return result

    @njit
    def _convolve_with_mask_2d(image, kernel, mask):
        """Determine the weighted average for each value."""
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
                        if not mask[ii, jj]:
                            jjj = wky + jj - j
                            num += kernel[iii, jjj] * image[ii, jj]
                            div += kernel[iii, jjj]
                if div:
                    result[i, j] = num / div
                else:
                    result[i, j] = np.nan
        return result

    @njit
    def _convolve_with_mask_3d(image, kernel, mask):
        """Determine the weighted average for each value."""
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
                                if not mask[ii, jj, kk]:
                                    kkk = wkz + kk - k
                                    num += (kernel[iii, jjj, kkk] *
                                            image[ii, jj, kk])
                                    div += kernel[iii, jjj, kkk]
                    if div:
                        result[i, j, k] = num / div
                    else:
                        result[i, j, k] = np.nan
        return result

    @njit
    def _convolve_with_mask_std_1d(image, kernel, mask, mean):
        """Determine the variance for each value."""
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
                if not mask[ii]:
                    iii = wkx + ii - i
                    # Difference to normal convolution.
                    diff = (image[ii] - mean[i]) ** 2
                    num += diff * kernel[iii]
                    div += kernel[iii]
            if div:
                result[i] = num / div
            else:
                result[i] = np.nan
        return result

    @njit
    def _convolve_with_mask_std_2d(image, kernel, mask, mean):
        """Determine the variance for each value."""
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
                        if not mask[ii, jj]:
                            jjj = wky + jj - j
                            # Difference to normal convolution.
                            diff = (image[ii, jj] - mean[i, j]) ** 2
                            num += diff * kernel[iii, jjj]
                            div += kernel[iii, jjj]
                if div:
                    result[i, j] = num / div
                else:
                    result[i, j] = np.nan
        return result

    @njit
    def _convolve_with_mask_std_3d(image, kernel, mask, mean):
        """Determine the variance for each value."""
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
                                if not mask[ii, jj, kk]:
                                    kkk = wkz + kk - k
                                    # Difference to normal convolution.
                                    diff = (image[ii, jj, kk] -
                                            mean[i, j, k]) ** 2
                                    num += diff * kernel[iii, jjj, kkk]
                                    div += kernel[iii, jjj, kkk]
                    if div:
                        result[i, j, k] = num / div
                    else:
                        result[i, j, k] = np.nan
        return result

    MODE_DIM_FUNC_MAP = {'convolution':   {1: _convolve_with_mask_1d,
                                           2: _convolve_with_mask_2d,
                                           3: _convolve_with_mask_3d},

                         'interpolation': {1: _interpolate_mask_1d,
                                           2: _interpolate_mask_2d,
                                           3: _interpolate_mask_3d},
                         }

    MODE_DIM_FUNC_MAP_STD = {'convolution':   {1: _convolve_with_mask_std_1d,
                                               2: _convolve_with_mask_std_2d,
                                               3: _convolve_with_mask_std_3d}
                             }

    @njit
    def insertionsort(items):
        """An insertion sort algorithm based on ``Numba``.

        Parameters
        ----------
        items : `numpy.ndarray`
            The items to be sorted.

        Returns
        -------
        nothing : `None`
            Sorting happens in-place!

        Notes
        -----
        The insertionsort is used to determine the median for the median-based
        convolution and interpolation. Most kernels tend to be small and
        insertionsort has low constant costs and outperforms normal median
        calculation in these cases.

        On my computer they break even only for more than 500 elements:

        .. code::

            from nddata.utils.numbautils import insertionsort
            from numba import njit
            import numpy as np

            @njit
            def numba_median(items):
                return np.median(items)

            # Let them be compiled for the timings
            numba_median(np.random.random(3))
            insertionsort(np.random.random(3))

            for i in [10, 20, 50, 100, 200, 300, 500, 1000]:
                data = np.random.random(i)
                print('-'*20 + str(i))
                # Copy the data so the constant costs are equal
                %timeit data = np.random.random(i); np.median(data)
                %timeit data = np.random.random(i); numba_median(data)
                %timeit data = np.random.random(i); insertionsort(data)
            --------------------10
            10000 loops, best of 3: 175 us per loop
            10000 loops, best of 3: 21.9 us per loop
            100000 loops, best of 3: 18.9 us per loop
            --------------------20
            10000 loops, best of 3: 174 us per loop
            10000 loops, best of 3: 22.9 us per loop
            100000 loops, best of 3: 19.4 us per loop
            --------------------50
            10000 loops, best of 3: 172 us per loop
            10000 loops, best of 3: 24.1 us per loop
            10000 loops, best of 3: 21.1 us per loop
            --------------------100
            10000 loops, best of 3: 173 us per loop
            10000 loops, best of 3: 26.5 us per loop
            10000 loops, best of 3: 26.6 us per loop
            --------------------200
            10000 loops, best of 3: 180 us per loop
            10000 loops, best of 3: 33.1 us per loop
            10000 loops, best of 3: 45.5 us per loop
            --------------------300
            10000 loops, best of 3: 186 us per loop
            10000 loops, best of 3: 39.2 us per loop
            10000 loops, best of 3: 72.8 us per loop
            --------------------500
            1000 loops, best of 3: 202 us per loop
            10000 loops, best of 3: 50.6 us per loop
            10000 loops, best of 3: 154 us per loop
            --------------------1000
            1000 loops, best of 3: 221 us per loop
            10000 loops, best of 3: 80 us per loop
            1000 loops, best of 3: 512 us per loop


        The constant costs of creating a random array do affect all these
        timings and obfuscate that for small arrays (size < 100) the
        insertionsort beats even the numba-jitted version of np.median.
        Therefore I chose the insertionsort appropach in the hope that noone
        would use large kernels for median filtering.
        """
        for i in range(items.size):
            tmp = items[i]
            k = i
            while k > 0 and tmp < items[k - 1]:
                items[k] = items[k - 1]
                k -= 1
            items[k] = tmp

    @njit
    def median_from_sorted(items):
        """Calculate the median from a sorted sequence.

        Parameters
        ----------
        items : `numpy.ndarray`
            The items to be sorted.

        Returns
        -------
        median : number
            The middle element if the items contain an odd number of elements
            or the average of the two middle elements if the length of items
            was even.

        Notes
        -----
        This function is meant to be complementary to the interpolation and
        convolution based on the median. The sorting is done by the
        ``insertionsort`` function (which seems to be faster than np.median)
        and this function calculates the median from this sorted array.

        .. note::
            The sorting must be done before it is not done inside this
            function.
        """
        # Median is the middle element (odd length)
        # or the mean of the two middle elements (even length)
        halfsize = items.size // 2
        if items.size % 2:
            return items[halfsize]
        else:
            return 0.5 * (items[halfsize - 1] + items[halfsize])

    @njit
    def _interpolate_median_mask_1d(image, kernel, mask):
        """Determine the median for each masked value."""
        nx = image.shape[0]
        nkx = kernel.shape[0]
        wkx = nkx // 2

        result = np.zeros(image.shape, dtype=np.float64)
        tmp = np.zeros(kernel.size, dtype=image.dtype)

        for i in range(0, nx, 1):
            if mask[i]:
                iimin = max(i - wkx, 0)
                iimax = min(i + wkx + 1, nx)
                elements = 0
                for ii in range(iimin, iimax, 1):
                    iii = wkx + ii - i
                    if not mask[ii] and kernel[iii]:
                        tmp[elements] = image[ii]
                        elements += 1
                if elements:
                    insertionsort(tmp[:elements])
                    median = median_from_sorted(tmp[:elements])
                    result[i] = median
                else:
                    result[i] = np.nan
            else:
                result[i] = image[i]
        return result

    @njit
    def _interpolate_median_mask_2d(image, kernel, mask):
        """Determine the median for each masked value."""
        nx = image.shape[0]
        ny = image.shape[1]
        nkx = kernel.shape[0]
        nky = kernel.shape[1]
        wkx = nkx // 2
        wky = nky // 2

        result = np.zeros(image.shape, dtype=np.float64)
        tmp = np.zeros(kernel.size, dtype=image.dtype)

        for i in range(0, nx, 1):
            for j in range(0, ny, 1):
                if mask[i, j]:
                    iimin = max(i - wkx, 0)
                    iimax = min(i + wkx + 1, nx)
                    jjmin = max(j - wky, 0)
                    jjmax = min(j + wky + 1, ny)
                    elements = 0
                    for ii in range(iimin, iimax, 1):
                        iii = wkx + ii - i
                        for jj in range(jjmin, jjmax, 1):
                            jjj = wky + jj - j
                            if not mask[ii, jj] and kernel[iii, jjj]:
                                tmp[elements] = image[ii, jj]
                                elements += 1
                    if elements:
                        insertionsort(tmp[:elements])
                        median = median_from_sorted(tmp[:elements])
                        result[i, j] = median
                    else:
                        result[i, j] = np.nan
                else:
                    result[i, j] = image[i, j]
        return result

    @njit
    def _interpolate_median_mask_3d(image, kernel, mask):
        """Determine the median for each masked value."""
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
        tmp = np.zeros(kernel.size, dtype=image.dtype)

        for i in range(0, nx, 1):
            for j in range(0, ny, 1):
                for k in range(0, nz, 1):
                    if mask[i, j, k]:
                        iimin = max(i - wkx, 0)
                        iimax = min(i + wkx + 1, nx)
                        jjmin = max(j - wky, 0)
                        jjmax = min(j + wky + 1, ny)
                        kkmin = max(k - wkz, 0)
                        kkmax = min(k + wkz + 1, nz)
                        elements = 0
                        for ii in range(iimin, iimax, 1):
                            iii = wkx + ii - i
                            for jj in range(jjmin, jjmax, 1):
                                jjj = wky + jj - j
                                for kk in range(kkmin, kkmax, 1):
                                    kkk = wkz + kk - k
                                    if (not mask[ii, jj, kk] and
                                            kernel[iii, jjj, kkk]):
                                        tmp[elements] = image[ii, jj, kk]
                                        elements += 1
                        if elements:
                            insertionsort(tmp[:elements])
                            median = median_from_sorted(tmp[:elements])
                            result[i, j, k] = median
                        else:
                            result[i, j, k] = np.nan
                    else:
                        result[i, j, k] = image[i, j, k]
        return result

    @njit
    def _convolve_median_with_mask_1d(image, kernel, mask):
        """Determine the median for each value."""
        nx = image.shape[0]
        nkx = kernel.shape[0]
        wkx = nkx // 2

        result = np.zeros(image.shape, dtype=np.float64)
        tmp = np.zeros(kernel.size, dtype=image.dtype)

        for i in range(0, nx, 1):
            iimin = max(i - wkx, 0)
            iimax = min(i + wkx + 1, nx)
            elements = 0
            for ii in range(iimin, iimax, 1):
                iii = wkx + ii - i
                if not mask[ii] and kernel[iii]:
                    tmp[elements] = image[ii]
                    elements += 1
            if elements:
                insertionsort(tmp[:elements])
                median = median_from_sorted(tmp[:elements])
                result[i] = median
            else:
                result[i] = np.nan
        return result

    @njit
    def _convolve_median_with_mask_2d(image, kernel, mask):
        """Determine the median for each value."""
        nx = image.shape[0]
        nkx = kernel.shape[0]
        wkx = nkx // 2
        ny = image.shape[1]
        nky = kernel.shape[1]
        wky = nky // 2

        result = np.zeros(image.shape, dtype=np.float64)
        tmp = np.zeros(kernel.size, dtype=image.dtype)

        for i in range(0, nx, 1):
            iimin = max(i - wkx, 0)
            iimax = min(i + wkx + 1, nx)
            for j in range(0, ny, 1):
                jjmin = max(j - wky, 0)
                jjmax = min(j + wky + 1, ny)
                elements = 0
                for ii in range(iimin, iimax, 1):
                    iii = wkx + ii - i
                    for jj in range(jjmin, jjmax, 1):
                        jjj = wky + jj - j
                        if not mask[ii, jj] and kernel[iii, jjj]:
                            tmp[elements] = image[ii, jj]
                            elements += 1
                if elements:
                    insertionsort(tmp[:elements])
                    median = median_from_sorted(tmp[:elements])
                    result[i, j] = median
                else:
                    result[i, j] = np.nan
        return result

    @njit
    def _convolve_median_with_mask_3d(image, kernel, mask):
        """Determine the median for each value."""
        nx = image.shape[0]
        nkx = kernel.shape[0]
        wkx = nkx // 2
        ny = image.shape[1]
        nky = kernel.shape[1]
        wky = nky // 2
        nz = image.shape[2]
        nkz = kernel.shape[2]
        wkz = nkz // 2

        result = np.zeros(image.shape, dtype=np.float64)
        tmp = np.zeros(kernel.size, dtype=image.dtype)

        for i in range(0, nx, 1):
            iimin = max(i - wkx, 0)
            iimax = min(i + wkx + 1, nx)
            for j in range(0, ny, 1):
                jjmin = max(j - wky, 0)
                jjmax = min(j + wky + 1, ny)
                for k in range(0, nz, 1):
                    kkmin = max(k - wkz, 0)
                    kkmax = min(k + wkz + 1, nz)

                    elements = 0

                    for ii in range(iimin, iimax, 1):
                        iii = wkx + ii - i
                        for jj in range(jjmin, jjmax, 1):
                            jjj = wky + jj - j
                            for kk in range(kkmin, kkmax, 1):
                                kkk = wkz + kk - k
                                if (not mask[ii, jj, kk] and
                                        kernel[iii, jjj, kkk]):
                                    tmp[elements] = image[ii, jj, kk]
                                    elements += 1
                    if elements:
                        insertionsort(tmp[:elements])
                        median = median_from_sorted(tmp[:elements])
                        result[i, j, k] = median
                    else:
                        result[i, j, k] = np.nan
        return result

    @njit
    def _convolve_median_with_mask_std_1d(image, kernel, mask, mean):
        """Determine the median absolute deviation for each value."""
        nx = image.shape[0]
        nkx = kernel.shape[0]
        wkx = nkx // 2

        result = np.zeros(image.shape, dtype=np.float64)
        tmp = np.zeros(kernel.size, dtype=image.dtype)

        for i in range(0, nx, 1):
            iimin = max(i - wkx, 0)
            iimax = min(i + wkx + 1, nx)
            elements = 0
            for ii in range(iimin, iimax, 1):
                iii = wkx + ii - i
                if not mask[ii] and kernel[iii]:
                    # Difference to normal convolution.
                    tmp[elements] = abs(image[ii] - mean[i])
                    elements += 1
            if elements:
                insertionsort(tmp[:elements])
                median = median_from_sorted(tmp[:elements])
                result[i] = median
            else:
                result[i] = np.nan
        return result

    @njit
    def _convolve_median_with_mask_std_2d(image, kernel, mask, mean):
        """Determine the median absolute deviation for each value."""
        nx = image.shape[0]
        nkx = kernel.shape[0]
        wkx = nkx // 2
        ny = image.shape[1]
        nky = kernel.shape[1]
        wky = nky // 2

        result = np.zeros(image.shape, dtype=np.float64)
        tmp = np.zeros(kernel.size, dtype=image.dtype)

        for i in range(0, nx, 1):
            iimin = max(i - wkx, 0)
            iimax = min(i + wkx + 1, nx)
            for j in range(0, ny, 1):
                jjmin = max(j - wky, 0)
                jjmax = min(j + wky + 1, ny)
                elements = 0
                for ii in range(iimin, iimax, 1):
                    iii = wkx + ii - i
                    for jj in range(jjmin, jjmax, 1):
                        jjj = wky + jj - j
                        if not mask[ii, jj] and kernel[iii, jjj]:
                            # Difference to normal convolution.
                            tmp[elements] = abs(image[ii, jj] - mean[i, j])
                            elements += 1
                if elements:
                    insertionsort(tmp[:elements])
                    median = median_from_sorted(tmp[:elements])
                    result[i, j] = median
                else:
                    result[i, j] = np.nan
        return result

    @njit
    def _convolve_median_with_mask_std_3d(image, kernel, mask, mean):
        """Determine the median absolute deviation for each value."""
        nx = image.shape[0]
        nkx = kernel.shape[0]
        wkx = nkx // 2
        ny = image.shape[1]
        nky = kernel.shape[1]
        wky = nky // 2
        nz = image.shape[2]
        nkz = kernel.shape[2]
        wkz = nkz // 2

        result = np.zeros(image.shape, dtype=np.float64)
        tmp = np.zeros(kernel.size, dtype=image.dtype)

        for i in range(0, nx, 1):
            iimin = max(i - wkx, 0)
            iimax = min(i + wkx + 1, nx)
            for j in range(0, ny, 1):
                jjmin = max(j - wky, 0)
                jjmax = min(j + wky + 1, ny)
                for k in range(0, nz, 1):
                    kkmin = max(k - wkz, 0)
                    kkmax = min(k + wkz + 1, nz)

                    elements = 0

                    for ii in range(iimin, iimax, 1):
                        iii = wkx + ii - i
                        for jj in range(jjmin, jjmax, 1):
                            jjj = wky + jj - j
                            for kk in range(kkmin, kkmax, 1):
                                kkk = wkz + kk - k
                                if (not mask[ii, jj, kk] and
                                        kernel[iii, jjj, kkk]):
                                    # Difference to normal convolution.
                                    tmp[elements] = abs(image[ii, jj, kk] -
                                                        mean[i, j, k])
                                    elements += 1
                    if elements:
                        insertionsort(tmp[:elements])
                        median = median_from_sorted(tmp[:elements])
                        result[i, j, k] = median
                    else:
                        result[i, j, k] = np.nan
        return result

    MODE_DIM_FUNC_MEDIAN_MAP = {
        'convolution':   {1: _convolve_median_with_mask_1d,
                          2: _convolve_median_with_mask_2d,
                          3: _convolve_median_with_mask_3d},

        'interpolation': {1: _interpolate_median_mask_1d,
                          2: _interpolate_median_mask_2d,
                          3: _interpolate_median_mask_3d},
        }

    MODE_DIM_FUNC_MEDIAN_STD_MAP = {
        'convolution':   {1: _convolve_median_with_mask_std_1d,
                          2: _convolve_median_with_mask_std_2d,
                          3: _convolve_median_with_mask_std_3d},
        }
