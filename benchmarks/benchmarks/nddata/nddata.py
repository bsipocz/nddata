# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

from nddata.nddata import NDData, StdDevUncertainty

import numpy as np


class NDDataInit:

    def setup(self):
        self.data = np.array((3, 3))
        self.flags = 1
        self.mask = 1
        self.wcs = 1
        self.unit = 'm'
        self.meta = {'1': 1}
        self.uncertainty = 10

    def time_data_nocopy(self):
        NDData(self.data, copy=False)

    def time_data_copy(self):
        NDData(self.data, copy=True)

    def time_all_nocopy(self):
        NDData(self.data, flags=self.flags, mask=self.mask, meta=self.meta,
               wcs=self.wcs, unit=self.unit, uncertainty=self.uncertainty,
               copy=False)

    def time_all_copy(self):
        NDData(self.data, flags=self.flags, mask=self.mask, meta=self.meta,
               wcs=self.wcs, unit=self.unit, uncertainty=self.uncertainty,
               copy=True)


class Arithmetic:

    def setup(self):
        mask = np.array([True, False, True])
        error = StdDevUncertainty([0.1, 0.1, 0.1], unit='mm')
        self.ndd1 = NDData(np.ones(3), unit='m', mask=mask, uncertainty=error)
        self.ndd2 = NDData(np.ones(3), unit='cm', mask=mask, uncertainty=error)

    def time_add(self):
        self.ndd1.add(self.ndd2)


class UncertaintyInit:

    def setup(self):
        self.data = np.ones(3)

    def time_data_nocopy(self):
        StdDevUncertainty(self.data, copy=False)

    def time_data_copy(self):
        StdDevUncertainty(self.data, copy=True)
