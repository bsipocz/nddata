# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

import numpy as np

from nddata.utils.stats import mode


class Mode:

    def setup(self):
        self.data = np.random.randint(0, 201, 20000)
        self.data2 = self.data / 100

    def time_mode_no_decimals(self):
        mode(self.data)

    def time_mode_decimals(self):
        mode(self.data2, decimals=2)
