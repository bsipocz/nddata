# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from astropy import log
from astropy.io import registry as io_registry
from astropy.io import fits
import astropy.units as u
from astropy.wcs import WCS

from ..nddata import NDData
from ..nduncertainty import StdDevUncertainty, UnknownUncertainty


__all__ = ['NDIOMixin', 'read_nddata_fits', 'write_nddata_fits']


class NDIOMixin(object):
    """Mixin class to connect `NDData` to the astropy input/output registry:
    `astropy.io.registry`.
    """

    @classmethod
    def read(cls, *args, **kwargs):
        """Read and parse gridded N-dimensional data and return as an
        `NDData`-derived object.

        This function provides the `NDDataBase` interface to the astropy
        unified I/O layer. This allows easily reading a file in the supported
        data formats.
        """
        return io_registry.read(cls, *args, **kwargs)

    def write(self, *args, **kwargs):
        """Write a gridded N-dimensional data object out in specified format.

        This function provides the `NDDataBase` interface to the astropy
        unified I/O layer. This allows easily writing a file in the supported
        data formats.
        """
        io_registry.write(self, *args, **kwargs)


def read_nddata_fits(filename, ext_data=0, ext_meta=0, ext_mask='mask',
                     ext_uncert='uncert', kw_unit='bunit',
                     dtype=None, **kwargs_for_open):
    """
    Read data from a FITS file and wrap the contents in a \
    `~astropy.nddata.NDData`.

    Parameters
    ----------
    filename : str and other types
        see :func:`~astropy.io.fits.open` what possible types are allowed.

    ext_data, ext_meta, ext_mask, ext_uncert : str or int, optional
        Extensions from which to read ``data``, ``meta``, ``mask`` and
        ``uncertainty``.
        Default is ``0`` (data), ``0`` (meta), ``'mask'`` (mask) and
        ``'uncert'`` (uncertainty).

    kw_unit : str or None, optional
        The header keyword which translates to the unit for the data. Set it
        to ``None`` if parsing the unit results in a ValueError during reading.
        Default is ``'bunit'``.

    dtype : `numpy.dtype`-like or None, optional
        If not ``None`` the data array is converted to this dtype before
        returning. See `numpy.ndarray.astype` for more details.
        Default is ``None``.

    kwargs_for_open :
        Additional keyword arguments that are passed to
        :func:`~astropy.io.fits.open` (not all of them might be possible).
    """

    # Hardcoded values to get additional information about mask and uncertainty
    kw_hdr_masktype = 'boolean mask'
    kw_hdr_uncerttype = {
        'standard deviation uncertainty': StdDevUncertainty,
        'unknown uncertainty type': UnknownUncertainty}

    with fits.open(filename, mode='readonly', **kwargs_for_open) as hdus:
        # Read the data and meta from the specified extensions
        data = hdus[ext_data].data
        if dtype is not None:
            data = data.astype(dtype)
        meta = hdus[ext_meta].header

        # Read the mask and uncertainty from the specified extensions but
        # silently fail if the extension does not exist.
        mask = None
        if ext_mask in hdus:
            mask = hdus[ext_mask].data
            # Convert it to boolean array?
            if kw_hdr_masktype in hdus[ext_mask].header.get('comment', []):
                mask = mask.astype(bool)

        uncertainty = None
        if ext_uncert in hdus:
            uncertainty = hdus[ext_uncert].data

            hdr = hdus[ext_uncert].header
            # Get the required class for the uncertainty
            cls = (kw_hdr_uncerttype[kw] for kw in kw_hdr_uncerttype
                   if kw in hdr.get('comment', []))
            cls = next(cls, UnknownUncertainty)

            # Get the unit for the uncertainty if present
            unit_ = hdr[kw_unit].lower() if kw_unit in hdr else None

            # Don't copy it here, if a copy is required do it when creating
            # NDData.
            uncertainty = cls(uncertainty, unit=unit_, copy=False)

        # Load unit and wcs from header
        unit = None
        if kw_unit is not None and kw_unit in meta:
            try:
                unit = u.Unit(meta[kw_unit])
            except ValueError as exc:
                log.info(str(exc))
                # TODO: Possibly convert it to lower-case and try it again.
                # Could yield totally wrong results if the prefix "M" would be
                # converted to "m".
                # Possible way to do it:
                # unit = u.Unit(meta[kw_unit].lower())
        wcs = WCS(meta)

    # Just create an NDData instance: This will be upcast to the appropriate
    # class
    return NDData(data, meta=meta, mask=mask, uncertainty=uncertainty,
                  wcs=wcs, unit=unit, copy=False)


def write_nddata_fits(ndd, filename, ext_mask='mask', ext_uncert='uncert',
                      kw_unit='bunit', **kwargs_for_write):
    """
    Take an `~astropy.nddata.NDData`-like object and save it as FITS file.

    Parameters
    ----------
    ndd : `astropy.nddata.NDData`-like
        The data which is to be saved. Must not be given when this function
        is called through the ``NDData.write``-method!

    filename : str
        The filename for the newly written file.

    ext_mask, ext_uncert : str or int, optional
        Extensions to which ``mask`` and ``uncertainty`` are written.
        Default is ``'mask'`` (mask) and ``'uncert'`` (uncertainty).

    kwargs_for_write :
        Additional keyword arguments that are passed to
        :func:`~astropy.io.fits.HDUList.writeto` (not all of them might be
        possible).

    Notes
    -----
    The ``data`` and ``meta`` are always written to the PrimaryHDU (extension
    number ``0``).
    """
    # Comment card strings to allow roundtripping (must be identical to read!)
    kw_hdr_masktype = 'boolean mask'
    kw_hdr_uncerttype = {
        StdDevUncertainty: 'standard deviation uncertainty',
        UnknownUncertainty: 'unknown uncertainty type'}

    # Copy or convert the meta to a FITS header
    if isinstance(ndd.meta, fits.Header):
        header = ndd.meta.copy()
    else:
        header = fits.Header(ndd.meta.items())

    # Update the (copied) header (unit, wcs)
    if ndd.unit is not None:
        header[kw_unit] = ndd.unit.to_string()
    elif kw_unit in header:
        del header[kw_unit]

    if ndd.wcs is not None:
        try:
            header.update(ndd.wcs.to_header())
        except AttributeError:
            # wcs has no to_header method
            # FIXME: Implement this if other wcs objects should be allowed.
            log.info("the wcs cannot be converted to header information.")

    # Create a HDUList containing data
    hdus = [fits.PrimaryHDU(ndd.data, header=header)]

    # And append mask to the HDUList (if present)
    try:
        # Convert mask to uint8 and set a keyword so that the opener knows
        # that it was a boolean mask and can convert it back again.
        if ndd.mask.dtype == 'bool':
            hdr = fits.Header()
            hdr.add_comment(kw_hdr_masktype)
            hdus.append(fits.ImageHDU(ndd.mask.astype(np.uint8), header=hdr,
                                      name=ext_mask))
        else:
            hdus.append(fits.ImageHDU(ndd.mask, name=ext_mask))
    except AttributeError:
        # Either no mask or mask had no dtype
        pass

    # And append the uncertainty (if present)
    try:
        # We need to save the uncertainty_type and the unit of the uncertainty
        # so that the uncertainty can be completly recovered.
        hdr = fits.Header()

        # Save the class of the uncertainty
        if ndd.uncertainty.__class__ in kw_hdr_uncerttype:
            hdr.add_comment(kw_hdr_uncerttype[ndd.uncertainty.__class__])

        # Save the unit of the uncertainty if it differs from the nddata
        # TODO: This comparison only works correctly for StdDevUncertainty...
        if ndd.uncertainty.unit != ndd.unit:
            hdr[kw_unit] = ndd.uncertainty.unit.to_string()

        hdus.append(fits.ImageHDU(ndd.uncertainty.array, header=hdr,
                                  name=ext_uncert))
    except AttributeError:
        # Either no uncertainty or no uncertainty array, unit or
        # uncertainty_type. Should not be possible because everything that
        # doesn't look like an NDUUncertainty is converted to one.
        pass

    # Convert to HDUList and write it to the file.
    with fits.HDUList(hdus) as hdulist:
        hdulist.writeto(filename, **kwargs_for_write)


# TODO: Register reader and writer WITHOUT identifier (for now...)
io_registry.register_reader('simple_fits', NDIOMixin, read_nddata_fits)
io_registry.register_writer('simple_fits', NDIOMixin, write_nddata_fits)
