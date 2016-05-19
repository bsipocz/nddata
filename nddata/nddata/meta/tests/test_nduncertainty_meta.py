# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from ..nduncertainty_meta import NDUncertaintyPropagatable


class PropagateableUncertainty(NDUncertaintyPropagatable):
    @property
    def supports_correlated(self):
        return super(PropagateableUncertainty, self).supports_correlated

    def propagate(self, *args, **kwargs):
        return None

    @property
    def uncertainty_type(self):
        return 'fun'


def test_no_correlated_by_default():
    assert PropagateableUncertainty().supports_correlated is False
