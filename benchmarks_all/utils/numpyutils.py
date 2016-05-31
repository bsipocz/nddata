# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

import numpy as np


class Pad:

    params = [2, 200]
    param_names = ['n']

    def setup(self, n):
        from nddata.utils.numpyutils import pad
        self.pad = pad
        self.data = np.ones((n, n), int)

    def time_pad(self, n):
        # TODO: Make this import at the top as soon as it's published
        # But for now the asv complains while discovering impossible imports
        # at top level...
        self.pad(self.data, ((2, 3), (2, 5)),
                 mode='constant', constant_values=0)
