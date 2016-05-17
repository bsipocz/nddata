0.1 (unreleased)
----------------

General
^^^^^^^

initial release


New Features
^^^^^^^^^^^^

 - ``utils.garbagecollector`` module added containing a function for testing
   for memory leaks: ``assert_memory_leak``. PR #4

 - ``utils.descriptors`` module added which contains several descriptors to
   allow reusing common attribute requirements. PR #13


API changes
^^^^^^^^^^^

 - ``nddata.utils`` is now a submodule. The affected utilities (``Cutout2D``,
   ...) must now be imported from there. PR #1

 - ``NDData`` implements the ``__astropy_nddata__`` interface during instance
   creation. PR #9

 - ``NDData`` does not accept non-numerical arrays as data argument anymore. PR #9


Bug fixes
^^^^^^^^^

 - None
