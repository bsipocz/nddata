# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from astropy import log
from astropy.io import registry as io_registry
from astropy.io import fits
from astropy.wcs import WCS

from ..nddata import NDDataBase
from ..nduncertainty_var import VarianceUncertainty
from ..nduncertainty_relstd import RelativeUncertainty
from ..nduncertainty_stddev import StdDevUncertainty
from ..nduncertainty_unknown import UnknownUncertainty


__all__ = ['NDIOMixin', 'read_nddata_fits', 'write_nddata_fits']


class NDIOMixin(object):
    """Mixin class to connect `~.NDDataBase` to the astropy input/output \
            registry: `astropy.io.registry`.
    """

    @classmethod
    def read(cls, *args, **kwargs):
        """Read and parse gridded N-dimensional data and return as an
        `~.NDDataBase`-derived object.

        This function provides the `~.NDDataBase` interface to the astropy
        unified I/O layer. This allows easily reading a file in the supported
        data formats.
        """
        return io_registry.read(cls, *args, **kwargs)

    def write(self, *args, **kwargs):
        """Write a gridded N-dimensional data object out in specified format.

        This function provides the `~.NDDataBase` interface to the astropy
        unified I/O layer. This allows easily writing a file in the supported
        data formats.
        """
        io_registry.write(self, *args, **kwargs)


# Header keywords reserved to save the mask dtype and uncertainty type
HDR_KEYWORD_MASK = 'NDDMASK'
HDR_KEYWORD_UNCERTAINTY = 'NDDERROR'
HDR_KEYWORD_UNIT = 'BUNIT'

HDR_VALUE_MASK = 'boolean'
# Header values for uncertainty mapped against the different classes of
# uncertainty present. Also an inverse Mapping so we can access them quickly.
HDR_VALUE_UNCERTAINTY = {StdDevUncertainty: 'stddev',
                         UnknownUncertainty: 'unknown',
                         VarianceUncertainty: 'variance',
                         RelativeUncertainty: 'relative'}
HDR_IVALUE_UNCERTAINTY = {
    val: key for key, val in HDR_VALUE_UNCERTAINTY.items()}


def read_nddata_fits(filename, ext_data=0, ext_meta=0, ext_mask='mask',
                     ext_uncert='uncert', ext_flags='flags', kw_unit='bunit',
                     parse_wcs=True, dtype=None, **kwargs_for_open):
    """ Read data from a FITS file and wrap the contents in a \
            `~nddata.nddata.NDDataBase`.

    Parameters
    ----------
    filename : str and other types
        see :func:`~astropy.io.fits.open` what possible types are allowed.

    ext_data, ext_meta, ext_mask, ext_uncert, ext_flags : str or int, optional
        Extensions from which to read ``data``, ``meta``, ``mask`` and
        ``uncertainty``.
        Default is ``0`` (data), ``0`` (meta), ``'mask'`` (mask) and
        ``'uncert'`` (uncertainty).

    kw_unit : str or None, optional
        The header keyword which translates to the unit for the data. Set it
        to ``None`` if parsing the unit results in a ValueError during reading.
        If this value in the header leads to an Exception while creating the
        class set it to ``None``.
        Default is ``'bunit'``.

    parse_wcs : `bool`, optional
        Try to create an `~astropy.wcs.WCS` object from the meta. If ``False``
        no attempt is made. This should only be set to ``False`` in case the
        resulting wcs is corrupt.
        Default is ``True``.

    dtype : `numpy.dtype`-like or None, optional
        If not ``None`` the data array is converted to this dtype before
        returning. See `numpy.ndarray.astype` for more details.
        Default is ``None``.

    kwargs_for_open :
        Additional keyword arguments that are passed to
        :func:`~astropy.io.fits.open` (not all of them might be possible).

    Returns
    -------
    ndd : `~nddata.nddata.NDDataBase`
        The wrapped FITS file contents.

    Notes
    -----
    It should also be able to process if the ``filename`` is an already loaded
    `~astropy.io.fits.HDUList`.
    """
    # It should also support if a HDUList is passed in here. The HDUList also
    # supports the context manager protocol so we open the file if it's not
    # a HDUList but if it's a HDUList we just use the HDUList as context
    # manager.
    isfile = not isinstance(filename, fits.HDUList)
    with (fits.open(filename, mode='readonly', **kwargs_for_open) if isfile
            else filename) as hdus:
        # Read the data and meta from the specified extensions
        data = hdus[ext_data].data
        if dtype is not None:
            data = data.astype(dtype)

        meta = hdus[ext_meta].header

        # Read in the specified extension for the mask and if it is present
        # use it as the mask for the output.
        mask = None
        if ext_mask in hdus:
            mask = hdus[ext_mask].data
            # Booleans cannot be saved by astropy.io.fits so we have set a flag
            # when writing that it should be a boolean mask. If the keyword
            # is in the header and the value of that keyword is the specified
            # value convert it to a boolean array. This should be that cautious
            # because we would otherwise loose information by downcasting the
            # dtype!
            maskhdr = hdus[ext_mask].header
            if HDR_VALUE_MASK == maskhdr.get(HDR_KEYWORD_MASK, None):
                mask = mask.astype(bool)

        # The same for the flags.
        flags = None
        if ext_flags in hdus:
            flags = hdus[ext_flags].data

        # and for the uncertainty
        uncertainty = None
        if ext_uncert in hdus:
            uncertainty = hdus[ext_uncert].data

            # The uncertainty could be one of several classes. Like with the
            # mask an appropriate flag should have been set during writing and
            # we now check if it present (defaulting to unknown if it isn't)
            # and then check if any uncertainty class is associated with the
            # corresponding value.
            hdr = hdus[ext_uncert].header
            cls = hdr.get(HDR_KEYWORD_UNCERTAINTY, 'unknown')
            cls = HDR_IVALUE_UNCERTAINTY.get(cls, UnknownUncertainty)

            # The uncertainty could also have a unit, check if it's present
            # and set and if so use it, otherwise let it be.
            unit_ = hdr[kw_unit] if kw_unit in hdr else None

            # Finally create the uncertainty by passing in the values and unit
            # WITHOUT making a copy here.
            uncertainty = cls(uncertainty, unit=unit_, copy=False)

        # Load unit and wcs from header, this could be problematic by
        # externally written files since the value of the header keyword might
        # not translate to an astropy.unit.Unit but if that's the case let it
        # fail later and the user needs to alter the FITS file manually or set
        # the kw_unit to None and set the unit afterwards.
        unit = meta[kw_unit] if kw_unit in meta else None

        # Create an astropy.wcs.WCS object from the meta associated with the
        # primary data. This could fail if the FITS file is invalid but then
        # one should choose to not create a WCS object.
        wcs = WCS(meta) if parse_wcs else None

    # Just create an NDData instance: This will be upcast to the appropriate
    # class
    return NDDataBase(data, meta=meta, mask=mask, uncertainty=uncertainty,
                      wcs=wcs, unit=unit, flags=flags, copy=False)


def write_nddata_fits(ndd, filename, ext_mask='mask', ext_uncert='uncert',
                      ext_flags='flags', **kwargs_for_write):
    """Take an `~nddata.nddata.NDDataBase`-like object and save it as FITS \
            file.

    Parameters
    ----------
    ndd : `~nddata.nddata.NDDataBase`-like
        The data which is to be saved. Must not be given when this function
        is called through the with
        :meth:`NDIOMixin.write`!

    filename : str
        The filename for the newly written file.

    ext_mask, ext_uncert, ext_flags : str or int, optional
        Extensions to which ``mask`` and ``uncertainty`` are written.
        Default is ``'mask'`` (mask) and ``'uncert'`` (uncertainty).

    kwargs_for_write :
        Additional keyword arguments that are passed to
        :meth:`~astropy.io.fits.HDUList.writeto` (not all of them might be
        possible).

    Notes
    -----
    The ``data`` and ``meta`` are always written to the PrimaryHDU (extension
    number ``0``).
    """
    # We will update the meta object with potentially altered WCS informations
    # or unit. We do not want that the instance is affected by this so we copy
    # if it is already a fits Header or implicitly copy it by casting it to a
    # Header object.
    if isinstance(ndd.meta, fits.Header):
        header = ndd.meta.copy()
    else:
        header = fits.Header(ndd.meta.items())

    # We need to insert the unit into the header potentially overwriting the
    # the previous set unit. But in case we explicitly removed the unit we
    # don't want any relict of the old unit to stay in the header. So we need
    # to remove old unit key-value pairs IF we don't overwrite them.
    if ndd.unit is not None:
        header[HDR_KEYWORD_UNIT] = ndd.unit.to_string()
    elif HDR_KEYWORD_UNIT in header:
        del header[HDR_KEYWORD_UNIT]

    # For now we assume that the WCS attribute is None or an astropy.wcs.WCS
    # object. In case the WCS is set we try updating the header information
    # in case anything was altered (maybe because of slicing).
    if ndd.wcs is not None:
        try:
            header.update(ndd.wcs.to_header())
        except AttributeError:
            # In case the wcs had no "to_header" we are not dealing with an
            # astropy.wcs.WCS object so we print an info-message and leave it
            # be.
            log.info("the wcs of type {0} cannot be used to update the header "
                     "and therefore the header remained unaffected."
                     "".format(ndd.wcs.__class__.__name__))

    # Create a HDUList containing data and header as primary data object.
    hdus = [fits.PrimaryHDU(ndd.data, header=header)]

    # Next try to append the mask.
    try:
        # If the mask is a boolean numpy array we need to convert it to the
        # lowest allowed dtype for FITS images: uint8 and update the header of
        # the mask extension so that we know when reading the file again that
        # the mask should be interpreted as boolean.
        if ndd.mask.dtype == 'bool':
            hdr = fits.Header()
            hdr[HDR_KEYWORD_MASK] = HDR_VALUE_MASK
            hdus.append(fits.ImageHDU(ndd.mask.astype(np.uint8), header=hdr,
                                      name=ext_mask))
        else:
            # In case it was no boolean array we just write it. No need to
            # special case anything here
            hdus.append(fits.ImageHDU(ndd.mask, name=ext_mask))
    except AttributeError:  # If there was no mask or mask had no dtype.
        pass

    # now the flags
    if ndd.flags is not None:
        hdus.append(fits.ImageHDU(ndd.flags, name=ext_flags))

    # and the uncertainty
    try:
        # We need to save the uncertainty_type and the unit of the uncertainty
        # so that the uncertainty can be completly recovered.
        hdr = fits.Header()

        # Save the class of the uncertainty
        cls_str = HDR_VALUE_UNCERTAINTY.get(ndd.uncertainty.__class__,
                                            'unknown')
        hdr[HDR_KEYWORD_UNCERTAINTY] = cls_str

        # Save the unit of the uncertainty if it differs from the nddata
        if ndd.uncertainty.unit is not None:
            hdr[HDR_KEYWORD_UNIT] = ndd.uncertainty.unit.to_string()

        hdus.append(fits.ImageHDU(ndd.uncertainty.data, header=hdr,
                                  name=ext_uncert))
    except AttributeError:  # no uncertainty or not NDUncertainty-like
        pass

    # Convert to HDUList and write it to the file.
    with fits.HDUList(hdus) as hdulist:
        hdulist.writeto(filename, **kwargs_for_write)


io_registry.register_reader('simple_fits', NDIOMixin, read_nddata_fits)
io_registry.register_writer('simple_fits', NDIOMixin, write_nddata_fits)
io_registry.register_identifier('simple_fits', NDIOMixin, fits.connect.is_fits)


def read_nddata_fits_header(filename, ext_meta=0, **kwargs_for_open):
    """Read header from a FITS file and wrap the contents in a \
            `~nddata.nddata.NDDataBase`.

    Parameters
    ----------
    filename : str and other types
        see :func:`~astropy.io.fits.open` what possible types are allowed.

    ext_meta : str or int, optional
        Extensions from which to read ``meta``.
        Default is ``0``.

    kwargs_for_open :
        Additional keyword arguments that are passed to
        :func:`~astropy.io.fits.open` (not all of them might be possible).

    Returns
    -------
    ndd : `~nddata.nddata.NDDataBase`
        The wrapped FITS file contents.

    Notes
    -----
    It should also be able to process if the ``filename`` is an already loaded
    `~astropy.io.fits.HDUList`.
    """
    # It should also support if a HDUList is passed in here. The HDUList also
    # supports the context manager protocol so we open the file if it's not
    # a HDUList but if it's a HDUList we just use the HDUList as context
    # manager.
    isfile = not isinstance(filename, fits.HDUList)
    with (fits.open(filename, mode='readonly', **kwargs_for_open) if isfile
            else filename) as hdus:

        meta = hdus[ext_meta].header

    # Just create an NDData instance: This will be upcast to the appropriate
    # class
    return NDDataBase(0, meta=meta, copy=False)

io_registry.register_reader('fits_header', NDIOMixin, read_nddata_fits_header)
