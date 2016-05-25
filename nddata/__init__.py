# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Affiliated packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *
# ----------------------------------------------------------------------------

# For egg_info test builds to pass, put package imports here.
if not _ASTROPY_SETUP_:
    # Import the dictionaries containing informations about the dependencies
    # and the version requirements.
    from deps import OPT_DEPS, MIN_VERSIONS
    from . import nddata
    from . import utils
