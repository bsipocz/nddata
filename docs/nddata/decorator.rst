*********************************************
Decorating functions to accept NDData objects
*********************************************

.. important:: The functionality described here is still experimental and will
               likely evolve over time as more packages make use of it.

Introduction
============

The `nddata.nddata` module includes a decorator
:func:`~nddata.nddata.utils.support_nddata` that makes it easy for developers
and users to write functions that can accept either
:class:`~nddata.nddata.NDDataBase` objects and also separate arguments.

Getting started
===============

Let's consider the following function::

    def func(data, wcs=None, unit=None, n_iterations=3):
        ...

Now let's say that we want to be able to call the function as ``func(nd)``
where ``nd`` is a :class:`~nddata.nddata.NDDataBase` instance. We can decorate
this function using :func:`~nddata.nddata.utils.support_nddata`::

    from nddata.nddata import support_nddata

    @support_nddata
    def func(data, wcs=None, unit=None, n_iterations=3):
        ...

which makes it so that when the user calls ``func(nd)``, the function would
automatically be called with::

    func(nd.data, wcs=nd.wcs, unit=nd.unit)

That is, the decorator looks at the signature of the function and checks if any
of the arguments are also properties of the :class:`~nddata.nddata.NDDataBase`
object, and passes them
as individual arguments. The function can also be called with separate
arguments as if it wasn't decorated.

An exception is raised if an :class:`~nddata.nddata.NDDataBase` property is set
but the function does not accept it - for example, if ``wcs`` is set, but the
function cannot support WCS objects, an error would be raised. On the other
hand, if an argument in the function does not exist in the
:class:`~nddata.nddata.NDDataBase` object or is  not set, it is simply left to
its default value.

If the function call succeeds, then the decorator returns the values from the
function unmodified by default. However, in some cases we may want to return
separate ``data``, ``wcs``, etc. if these were passed in separately, and a new
:class:`~nddata.nddata.NDDataBase` instance otherwise. To do this, you can
specify ``repack=True`` in the decorator and provide a list of the names of the
output arguments from the function::

    @support_nddata(repack=True, returns=['data', 'wcs'])
    def func(data, wcs=None, unit=None, n_iterations=3):
        ...

With this, the function will return separate values if ``func`` is called with
separate arguments, and an object with the same class type as the input if the
input is an :class:`~nddata.nddata.NDDataBase` or subclass instance.

Finally, the decorator can be made to restrict input to specific
:class:`~nddata.nddata.NDDataBase` sub-classes (and sub-classes of those) using
the ``accepts`` option::

    @support_nddata(accepts=CCDImage)
    def func(data, wcs=None, unit=None, n_iterations=3):
        ...

