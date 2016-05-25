# Licensed under a 3-clause BSD style license - see LICENSE.rst


from distutils.version import LooseVersion


__all__ = ['OPT_DEPS', 'MIN_VERSIONS']


def cmp_version(actual, ref):
    """Function that uses `distutils.version.LooseVersion` to compare if \
            the installed version is at least the reference version.


    The input parameters can be anything that can be passed to
    LooseVersion. For example ``cmp_version(np.__version, '1.9')`` will
    return ``True`` if the installed numpy version is at least 1.9 or
    ``False`` otherwise.
    """
    return LooseVersion(actual) >= LooseVersion(ref)

# Determine which optional dependencies are installed.
# TODO: Switch to default-dict later so that unregistered packages return
# zero by default and not a KeyError.
OPT_DEPS = {}

try:
    import numba as nb
    OPT_DEPS['NUMBA'] = True
except ImportError:
    OPT_DEPS['NUMBA'] = False

try:
    import scipy
    OPT_DEPS['SCIPY'] = True
except ImportError:
    OPT_DEPS['SCIPY'] = False

try:
    import numba as nb
    OPT_DEPS['NUMBA'] = True
except ImportError:
    OPT_DEPS['NUMBA'] = False

try:
    import skimage
    OPT_DEPS['SCIKIT-IMAGE'] = True
except ImportError:
    OPT_DEPS['SCIKIT-IMAGE'] = False

# Some functions or tests require at least a minimum version, I'll register
# them here so we don't need to do it in the subpackages and I have more
# overview and consistent names:
# TODO: Make this a default dict or something that is calculated on the
# fly.
MIN_VERSIONS = {}

import numpy as np
import astropy

MIN_VERSIONS['NUMPY_1_9'] = cmp_version(np.__version__, '1.9')
MIN_VERSIONS['NUMPY_1_10'] = cmp_version(np.__version__, '1.10')

MIN_VERSIONS['ASTROPY_1_2'] = cmp_version(astropy.__version__, '1.2')
