# Licensed under a 3-clause BSD style license - see LICENSE.rst

from .. import NDUncertaintyMeta


def test_is_metaclass():
    # Create a minimal subclass that implements all abstract properties and
    # methods of NDUncertaintyMeta

    class MinimalSubclass(NDUncertaintyMeta):
        def __init__(self):
            super(MinimalSubclass, self).__init__()

        @property
        def data(self):
            return super(MinimalSubclass, self).data

        @property
        def unit(self):
            return super(MinimalSubclass, self).unit

        @property
        def uncertainty_type(self):
            return super(MinimalSubclass, self).uncertainty_type

    ndd = MinimalSubclass()
    for attr in ('data', 'unit', 'uncertainty_type'):
        assert getattr(ndd, attr) is None
