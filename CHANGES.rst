0.1 (unreleased)
----------------

General
^^^^^^^

initial release


New Features
^^^^^^^^^^^^

 - ``utils.garbagecollector`` module added containing a function for testing
   for memory leaks: ``assert_memory_leak``. PR #4


API changes
^^^^^^^^^^^

 - ``nddata.utils`` is now a submodule. The affected utilities (``Cutout2D``,
   ...) must now be imported from there. PR #1


Bug fixes
^^^^^^^^^

 - None
