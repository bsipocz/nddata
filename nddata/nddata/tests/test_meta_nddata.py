# Licensed under a 3-clause BSD style license - see LICENSE.rst

from .. import NDDataMeta


def test_is_metaclass():
    # Create a minimal subclass that implements all abstract properties and
    # methods of NDDataMeta

    class MinimalSubclass(NDDataMeta):
        def __init__(self):
            super(MinimalSubclass, self).__init__()

        @property
        def data(self):
            return super(MinimalSubclass, self).data

        @property
        def mask(self):
            return super(MinimalSubclass, self).mask

        @property
        def unit(self):
            return super(MinimalSubclass, self).unit

        @property
        def wcs(self):
            return super(MinimalSubclass, self).wcs

        @property
        def meta(self):
            return super(MinimalSubclass, self).meta

        @property
        def uncertainty(self):
            return super(MinimalSubclass, self).uncertainty

    ndd = MinimalSubclass()
    for attr in ('data', 'meta', 'uncertainty', 'wcs', 'mask', 'unit'):
        assert getattr(ndd, attr) is None
