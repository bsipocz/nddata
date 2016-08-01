# Licensed, see
# http://www.lpl.arizona.edu/~ianc/python/_modules/spec.html#optimalExtract

import numpy as np


__all__ = ['optimalExtract', 'DER_SNR']


def DER_SNR(flux):
    """Calculate the Signal-to-noise-ratio based on median absolute deviation.

    Parameters
    ----------
    flux : `numpy.ndarray`-like
        An array containing the measured flux. Should contain more than 4
        elements otherwise a snr of 0 is returned.

    Returns
    -------
    snr : number
        The calculated signal-to-noise ratio.

    Examples
    --------
    >>> import numpy as np
    >>> from nddata.utils.extractionction import DER_SNR

    >>> DER_SNR([1,10,2,5])
    0.0

    >>> np.random.seed(5678)
    >>> DER_SNR(np.random.normal(100,2,1000))

    >>> np.random.seed(None)

    .. note::
        Taken from http://www.stecf.org/software/ASTROsoft/DER_SNR/der_snr.py
        Author: Felix Stoehr, ST-ECF
    """
    flux = np.asarray(flux)

    # Values that are exactly zero (padded) are skipped
    flux = np.asarray(flux[np.where(flux != 0.0)])
    n = len(flux)

    # For spectra shorter than this, no value can be returned
    if (n > 4):
        signal = np.median(flux)
        noise = 0.6052697 * np.median(abs(2.0 * flux[2:n-2] -
                                          flux[0:n-4] - flux[4:n]))
        return signal / noise

    else:
        return 0.0


def optimalExtract(*args, **kw):
    """Extract spectrum, following Horne 1986.

    Parameters
    ----------
    data : 2D `numpy.ndarray`
        Appropriately calibrated frame from which to extract
        spectrum.  Should be in units of ADU, not electrons!

    variance : 2D `numpy.ndarray`
        Variances of pixel values in 'data'.

    gain : number
        Detector gain, in electrons per ADU.

    readnoise : number
        Detector readnoise, in electrons.

    goodpixelmask : 2D `numpy.ndarray`, optional
        Equals 0 for bad pixels, 1 for good pixels.
        Default is ``numpy.ones(data.shape)``.

    bkg_radii : 2- or 4-sequence, optional
        If length 2: inner and outer radii to use in computing
        background. Note that for this to be effective, the spectral
        trace should be positions in the center of 'data.'

        If length 4: start and end indices of both apertures for
        background fitting, of the form [b1_start, b1_end, b2_start,
        b2_end] where b1 and b2 are the two background apertures, and
        the elements are arranged in strictly ascending order.
        Default is ``[15, 20]``.

    extract_radius : `int` or 2-sequence, optional
        radius to use for both flux normalization and extraction. If
        a sequence, the first and last indices of the array to use
        for spectral normalization and extraction.
        Default is ``10``.

    dispaxis : `bool`, optional
        1 for horizontal spectrum, 0 for vertical spectrum.
        Default is ``0``.

    bord : `int` >= 0, optional
        Degree of polynomial background fit.
        Default is ``0``.

    bsigma : `int` >= 0, optional
        Sigma-clipping threshold for computing background.
        Default is ``3``.

    pord : `int` >= 0, optional
        Degree of polynomial fit to construct profile.
        Default is ``2``.

    psigma : `int` >= 0, optional
        Sigma-clipping threshold for computing profile.
        Default is ``4``.

    csigma : `int` >= 0, optional
        Sigma-clipping threshold for cleaning & cosmic-ray rejection.
        Default is ``5``.

    finite : `bool`, optional
        If true, mask all non-finite values as bad pixels.
        Default is ``True``.

    nreject : `int` > 0, optional
        Number of pixels to reject in each iteration.
        Default is ``100``.

    Returns
    -------
    spectrum_flux : `numpy.ndarray`
        spectrum flux (in electrons).

    uncertainty : `numpy.ndarray`
        uncertainty on spectrum flux.

    background_flux : `numpy.ndarray`
        background flux.

    good_pixelmask : `numpy.ndarray`
        good pixels after rejection.

    Notes
    -----
    Horne's classic optimal extraction algorithm is optimal only so
    long as the spectral traces are very nearly aligned with
    detector rows or columns.  It is *not* well-suited for
    extracting substantially tilted or curved traces, for the
    reasons described by Marsh 1989, Mukai 1990.

    .. note::
        This function is taken from:
        http://www.lpl.arizona.edu/~ianc/python/_modules/spec.html#optimalExtract
        and slightly adapted to work with newer numpy features and python 3.
    """
    # Parse inputs:
    frame, variance, gain, readnoise = args[0:4]

    # Parse options:
    if 'goodpixelmask' in kw:
        goodpixelmask = np.array(kw['goodpixelmask'], dtype=bool, copy=True)
    else:
        goodpixelmask = np.ones(frame.shape, dtype=bool)

    if 'dispaxis' in kw:
        if kw['dispaxis'] == 1:
            frame = frame.transpose()
            variance = variance.transpose()
            goodpixelmask = goodpixelmask.transpose()

    verbose = kw.get('verbose', False)
    bkg_radii = kw.get('bkg_radii', [15, 20])
    extract_radius = kw.get('extract_radius', 10)
    bord = kw.get('bord', 1)
    bsigma = kw.get('bsigma', 3)
    pord = kw.get('pord', 2)
    psigma = kw.get('psigma', 4)
    csigma = kw.get('csigma', 5)
    finite = kw.get('finite', True)
    nreject = kw.get('nreject', 100)

    if finite:
        goodpixelmask &= np.isfinite(frame) & np.isfinite(variance)

    variance[~goodpixelmask] = frame[goodpixelmask].max() * 1e9
    nlam, fitwidth = frame.shape

    xxx = np.arange(-fitwidth/2, fitwidth/2)
    xxx0 = np.arange(fitwidth)
    # Set all borders of background aperture:
    if len(bkg_radii) == 4:
        backgroundAperture = (
            ((xxx0 > bkg_radii[0]) & (xxx0 <= bkg_radii[1])) |
            ((xxx0 > bkg_radii[2]) & (xxx0 <= bkg_radii[3])))
    # Assume trace is centered, and use only radii.
    else:
        backgroundAperture = ((np.abs(xxx) > bkg_radii[0]) &
                              (np.abs(xxx) <= bkg_radii[1]))

    if hasattr(extract_radius, '__iter__'):
        extractionAperture = ((xxx0 > extract_radius[0]) &
                              (xxx0 <= extract_radius[1]))
    else:
        extractionAperture = np.abs(xxx) < extract_radius

    nextract = extractionAperture.sum()
    xb = xxx[backgroundAperture]

    # Step3: Sky Subtraction
    if bord == 0:
        # faster to take weighted mean:
        background = np.average(frame[:, backgroundAperture],
                                weights=(goodpixelmask/variance)[:, backgroundAperture],
                                axis=1)
    else:
        background = 0. * frame
        for ii in range(nlam):
            _w = (goodpixelmask / variance)[ii, backgroundAperture]
            fit = polyfitr(xb, frame[ii, backgroundAperture], bord, bsigma,
                           w=_w, verbose=verbose-1)
            background[ii, :] = np.polyval(fit, xxx)

    # (my 3a: mask any bad values)
    badBackground = True - np.isfinite(background)
    background[badBackground] = 0.
    if verbose and badBackground.any():
        print("Found bad background values at: ", badBackground.nonzero())

    skysubFrame = frame - background

    # Step4: Extract 'standard' spectrum and its variance
    standardSpectrum = np.expand_dims(
        nextract * np.average(skysubFrame[:, extractionAperture],
                              weights=goodpixelmask[:, extractionAperture],
                              axis=1), axis=1)
    varStandardSpectrum = np.expand_dims(
        nextract * np.average(variance[:, extractionAperture],
                              weights=goodpixelmask[:, extractionAperture],
                              axis=1), axis=1)

    # (my 4a: mask any bad values)
    badSpectrum = ~np.isfinite(standardSpectrum)
    standardSpectrum[badSpectrum] = 1.
    varStandardSpectrum[badSpectrum] = (
        varStandardSpectrum[~badSpectrum].max() * 1e9)

    # Step5: Construct spatial profile; enforce positivity & normalization
    normData = skysubFrame / standardSpectrum
    varNormData = variance / standardSpectrum**2

    # Iteratively clip outliers
    newBadPixels = True
    iter = -1
    if verbose:
        print("Looking for bad pixel outliers in profile construction.")
    xl = np.linspace(-1., 1., nlam)

    while newBadPixels:
        iter += 1

        if pord == 0:
            # faster to take weighted mean:
            profile = np.tile(np.average(normData, weights=(goodpixelmask / varNormData),
                                         axis=0), (nlam, 1))
        else:
            profile = 0. * frame
            for ii in range(fitwidth):
                fit = polyfitr(xl, normData[:, ii], pord, np.inf,
                               w=(goodpixelmask/varNormData)[:, ii],
                               verbose=verbose-1)
                profile[:, ii] = np.polyval(fit, xl)

        if profile.min() < 0:
            profile[profile < 0] = 0.
        profile /= profile.sum(1).reshape(nlam, 1)

        # Step6: Revise variance estimates
        modelData = standardSpectrum * profile + background
        variance = np.abs(modelData)/gain + (readnoise/gain)**2
        variance /= goodpixelmask + 1e-9
        # Avoid infinite variance

        outlierSigmas = (frame - modelData)**2 / variance
        if outlierSigmas.max() > psigma**2:
            maxRejectedValue = max(
                psigma**2, np.sort(outlierSigmas[:, extractionAperture].ravel())[-nreject])
            worstOutliers = (outlierSigmas >= maxRejectedValue).nonzero()
            goodpixelmask[worstOutliers] = False
            newBadPixels = True
            numberRejected = len(worstOutliers[0])
        else:
            newBadPixels = False
            numberRejected = 0

        if verbose:
            print("Rejected %i pixels on this iteration " % numberRejected)

        # Step5: Construct spatial profile; enforce positivity & normalization
        varNormData = variance / standardSpectrum**2

    if verbose:
        print("%i bad pixels found" % iter)

    # Iteratively clip Cosmic Rays
    newBadPixels = True
    iter = -1
    if verbose:
        print("Looking for bad pixel outliers in optimal extraction.")
    while newBadPixels:
        iter += 1

        # Step 8: Extract optimal spectrum and its variance
        gp = goodpixelmask * profile
        denom = (gp * profile / variance)[:, extractionAperture].sum(1)
        spectrum = ((gp * skysubFrame / variance)[:, extractionAperture].sum(1) / denom)
        spectrum = spectrum.reshape(nlam, 1)
        varSpectrum = (gp[:, extractionAperture].sum(1) / denom)
        varSpectrum = varSpectrum.reshape(nlam, 1)

        # Step6: Revise variance estimates
        modelData = spectrum * profile + background
        variance = np.abs(modelData)/gain + (readnoise/gain)**2
        # Avoid infinite variance
        variance /= goodpixelmask + 1e-9

        # Iterate until worse outliers are all identified:
        outlierSigmas = (frame - modelData)**2/variance
        if outlierSigmas.max() > csigma**2:
            maxRejectedValue = max(csigma**2, np.sort(outlierSigmas[:, extractionAperture].ravel())[-nreject])
            worstOutliers = (outlierSigmas >= maxRejectedValue).nonzero()
            goodpixelmask[worstOutliers] = False
            newBadPixels = True
            numberRejected = len(worstOutliers[0])
        else:
            newBadPixels = False
            numberRejected = 0

        if verbose:
            print("Rejected %i pixels on this iteration ", numberRejected)

    if verbose:
        print("%i bad pixels found" % iter)

    ret = (spectrum, varSpectrum, profile, background, goodpixelmask)

    return ret


def wmean(a, w, axis=None, reterr=False):
    """wmean(a, w, axis=None)

    Perform a weighted mean along the specified axis.

    Parameters
    ----------
    a : sequence or `numpy.ndarray`
        data for which weighted mean is computed

    w : sequence or `numpy.ndarray`
        weights of data -- e.g., 1./sigma^2

    axis : `int` or `None`, optional
        axis along which to compute the weighted mean.

    reterr : `bool`, optional
        If True, return the tuple (mean, err_on_mean), where
        err_on_mean is the unbiased estimator of the sample standard
        deviation.
        Default is ``False``.

    Returns
    -------
    wmean : `numpy.ndarray` or scalar
        weighted mean

    err : `numpy.ndarray` or scalar
        error of the weighted mean. Only returned if ``reterr`` is True.
    """

    newdata = np.array(a, subok=True, copy=True)
    newweights = np.array(w, subok=True, copy=True)

    if axis is None:
        newdata = newdata.ravel()
        newweights = newweights.ravel()
        axis = 0

    ash = list(newdata.shape)
    wsh = list(newweights.shape)

    nsh = list(ash)
    nsh[axis] = 1

    if ash != wsh:
        raise ValueError('Data and weight must be arrays of same shape.')

    wsum = newweights.sum(axis=axis).reshape(nsh)

    weightedmean = (a * newweights).sum(axis=axis).reshape(nsh) / wsum
    if reterr:
        # Biased estimator:
        # e_weightedmean = sqrt(
        #   (newweights * (a - weightedmean)**2).sum(axis=axis) / wsum)

        # Unbiased estimator:
        # e_weightedmean = sqrt(
        #   (wsum / (wsum**2 - (newweights**2).sum(axis=axis))) *
        #   (newweights * (a - weightedmean)**2).sum(axis=axis))

        # Standard estimator:
        e_weightedmean = np.sqrt(1. / newweights.sum(axis=axis))

        ret = weightedmean, e_weightedmean
    else:
        ret = weightedmean

    return ret


def polyfitr(x, y, N, s, fev=100, w=None, diag=False, clip='both',
             verbose=False, plotfit=False, plotall=False, eps=1e-13,
             catchLinAlgError=False):
    """Matplotlib's polyfit with weights and sigma-clipping rejection.

    Do a best fit polynomial of order N of y to x.  Points whose fit
    residuals exeed s standard deviations are rejected and the fit is
    recalculated.  Return value is a vector of polynomial
    coefficients [pk ... p1 p0].

    Parameters
    ----------
    w :
        a set of weights for the data.

    fev :
        number of function evaluations to call before stopping.

    diag :
        'diag'nostic flag:  Return the tuple (p, chisq, n_iter).

    clip:
        Do clipping according to:

        - 'both' -- remove outliers +/- 's' sigma from fit
        - 'above' -- remove outliers 's' sigma above fit
        - 'below' -- remove outliers 's' sigma below fit

    catchLinAlgError : bool
        If True, don't bomb on LinAlgError; instead, return [0, 0, ... 0].

    Notes
    -----
    Iterates so long as n_newrejections > 0 AND n_iter < fev.
    """
    xx = np.asarray(x)
    yy = np.asarray(y)
    noweights = (w is None) or (w.size == 0)
    if noweights:
        ww = np.ones(xx.shape, float)
    else:
        ww = np.asarray(w)

    ii = 0
    nrej = 1

    if noweights:
        goodind = np.isfinite(xx) * np.isfinite(yy)
    else:
        goodind = np.isfinite(xx) * np.isfinite(yy) * np.isfinite(ww)

    xx2 = xx[goodind]
    yy2 = yy[goodind]
    ww2 = ww[goodind]

    while (ii < fev and (nrej != 0)):
        if noweights:
            p = np.polyfit(xx2, yy2, N)
            residual = yy2 - np.polyval(p, xx2)
            stdResidual = np.std(residual)
            clipmetric = s * stdResidual
        else:
            if catchLinAlgError:
                try:
                    p = np.polyfit(xx2, yy2, N, w=ww2)
                except np.linalg.LinAlgError:
                    p = np.zeros(N+1, dtype=float)
            else:
                p = np.polyfit(xx2, yy2, N, w=ww2)

            residual = (yy2 - np.polyval(p, xx2)) * np.sqrt(ww2)
            clipmetric = s

        if clip == 'both':
            worstOffender = abs(residual).max()
            if worstOffender <= clipmetric or worstOffender < eps:
                ind = np.ones(residual.shape, dtype=bool)
            else:
                ind = abs(residual) < worstOffender
        elif clip == 'above':
            worstOffender = residual.max()
            if worstOffender <= clipmetric:
                ind = np.ones(residual.shape, dtype=bool)
            else:
                ind = residual < worstOffender
        elif clip == 'below':
            worstOffender = residual.min()
            if worstOffender >= -clipmetric:
                ind = np.ones(residual.shape, dtype=bool)
            else:
                ind = residual > worstOffender
        else:
            ind = np.ones(residual.shape, dtype=bool)

        xx2 = xx2[ind]
        yy2 = yy2[ind]
        if not noweights:
            ww2 = ww2[ind]
        ii = ii + 1
        nrej = len(residual) - len(xx2)

    if diag:
        chisq = ((residual)**2 / yy2).sum()
        p = (p, chisq, ii)

    return p
