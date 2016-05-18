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

 - ``utils.sentinels`` module added to create custom sentinals. PR #17

 - NDData got a copy method. PR #18


API changes
^^^^^^^^^^^

 - ``nddata.utils`` is now a submodule. The affected utilities (``Cutout2D``,
   ...) must now be imported from there. PR #1

 - ``NDData`` implements the ``__astropy_nddata__`` interface during instance
   creation. PR #9

 - ``NDData`` does not accept non-numerical arrays as data argument anymore. PR #9


Bug fixes
^^^^^^^^^

 - Given explicit and implicit attributes during NDData creation the implicit
   one was ignored if the implicit attribute was None and the explicit argument
   was not None. PR #17
