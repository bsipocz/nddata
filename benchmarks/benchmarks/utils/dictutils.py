# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

import random
import string
import operator

from nddata.utils.dictutils import (dict_merge, dict_merge_keep_all,
                                    dict_merge_keep_all_fill_missing)


class Merge:

    def setup(self):
        # Seed random numbers so it doesn't depend on randomness how fast the
        # function is. random is only used so I don't need to write the
        # dictionaries.
        self.dicts = [dict([(k, v)
                      for k in random.sample(string.ascii_letters, 20)])
                      for v in range(100)]

    def time_last(self):
        # Without fold func normal dict updates are used.
        dict_merge(*self.dicts)

    def time_custom(self):
        # Adding the values seems appropriate here since it doesn't
        # discriminatefor any structure in the data.
        dict_merge(*self.dicts, foldfunc=operator.add)

    def time_keepall(self):
        # The other function for merging but keeping all encountered values.
        dict_merge_keep_all(*self.dicts)

    def time_keepall_fill(self):
        # And the last merging function that keeps all values but also fills
        # in missing values.
        dict_merge_keep_all_fill_missing(*self.dicts)
