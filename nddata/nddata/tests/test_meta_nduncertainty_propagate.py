# Licensed under a 3-clause BSD style license - see LICENSE.rst

from .. import NDUncertaintyPropagationMeta, NDUncertaintyMeta


def test_is_metaclass():
    # Create a minimal subclass that implements all abstract properties and
    # methods of NDUncertaintyMeta

    class MinimalSubclass(NDUncertaintyPropagationMeta, NDUncertaintyMeta):
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

        @property
        def supports_correlated(self):
            return super(MinimalSubclass, self).supports_correlated

        @property
        def parent_nddata(self):
            return super(MinimalSubclass, self).parent_nddata

        def propagate(self, operation, other, result=None, correlation=None):
            return super(MinimalSubclass, self).propagate(operation, other,
                                                          result, correlation)

    ndd = MinimalSubclass()
    for attr in ('data', 'unit', 'uncertainty_type', 'parent_nddata',
                 'supports_correlated'):
        assert getattr(ndd, attr) is None

    assert ndd.propagate(None, None, None, None) is None
