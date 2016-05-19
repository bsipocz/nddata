# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import collections
from fractions import Fraction
from decimal import Decimal

import numpy as np

from astropy.tests.helper import pytest
from astropy.io import fits
import astropy.units as u

from .. import descriptors
from ..descriptors import BaseDescriptor, AdvancedDescriptor


def test_basedescriptor():

    class BaseDescriptorTest(object):
        attr_direct1 = BaseDescriptor('attr_direct1', doc="""Not funny!""",
                                      copy=False)
        attr_direct2 = BaseDescriptor('attr_direct2', doc="", copy=False)
        attr_direct3 = BaseDescriptor('attr_direct3', doc="""Not funny!""",
                                      copy=True)
        attr_direct4 = BaseDescriptor('attr_direct4', copy=True)

        attr_direct5 = BaseDescriptor('wrong')

        @BaseDescriptor
        def attr_indirect1(self):
            """Not funny!"""

        @BaseDescriptor
        def attr_indirect2(self):
            pass

    # test docstrings
    assert not BaseDescriptorTest.attr_direct2.__doc__
    assert not BaseDescriptorTest.attr_direct4.__doc__
    assert not BaseDescriptorTest.attr_indirect2.__doc__
    assert BaseDescriptorTest.attr_direct1.__doc__ == 'Not funny!'
    assert BaseDescriptorTest.attr_direct3.__doc__ == 'Not funny!'
    assert BaseDescriptorTest.attr_indirect1.__doc__ == 'Not funny!'

    # test getter without set attribute
    t = BaseDescriptorTest()
    for i in ('attr_direct1', 'attr_direct2', 'attr_direct3', 'attr_direct4',
              'attr_indirect1', 'attr_indirect2'):
        assert not hasattr(t, '_' + i)
        assert getattr(t, i) is None
        assert not hasattr(t, '_' + i)

        # set each attribute
        setattr(t, i, 5)
        assert getattr(t, i) == 5
        assert getattr(t, '_' + i) == 5

        # delete each attribute
        delattr(t, i)
        assert getattr(t, i) is None
        assert not hasattr(t, '_' + i)

    # test copy
    data = [1, 2, 3]
    t.attr_direct1 = data
    t.attr_indirect1 = data
    t.attr_direct3 = data
    data[0] = 10
    assert t.attr_direct1[0] == 10     # reference copy=False explicit
    assert t.attr_indirect1[0] == 10   # reference copy=False implicit
    assert t.attr_direct3[0] == 1      # this copied

    # test the descriptor where the public and private name differ
    assert not hasattr(t, '_wrong')
    assert not hasattr(t, '_attr_direct5')
    t.attr_direct5 is None
    assert not hasattr(t, '_wrong')
    assert not hasattr(t, '_attr_direct5')

    t.attr_direct5 = 5
    assert t.attr_direct5 == 5
    assert t._wrong == 5
    assert not hasattr(t, '_attr_direct5')

    del t.attr_direct5
    t.attr_direct5 is None
    assert not hasattr(t, '_wrong')
    assert not hasattr(t, '_attr_direct5')


def test_advanceddescriptor():

    class AdvancedDescriptorTest(object):
        data1 = AdvancedDescriptor('data1', copy=False)
        data2 = AdvancedDescriptor('data2', doc="", copy=True)
        datax = AdvancedDescriptor('wrong', doc="", copy=False)

        @AdvancedDescriptor
        def data3(self):
            pass

    t = AdvancedDescriptorTest()

    # test no private attribute is set
    assert not hasattr(t, '_data1')
    assert not hasattr(t, '_data2')
    assert not hasattr(t, '_data3')
    assert not hasattr(t, '_datax')
    assert not hasattr(t, '_wrong')

    # test getter without private attribute returns None
    assert t.data1 is None
    assert t.data2 is None
    assert t.data3 is None
    assert t.datax is None

    # Test after first get call the attribute is set to default
    assert t._data1 is None
    assert t._data2 is None
    assert t._data3 is None
    assert t._wrong is None
    assert not hasattr(t, '_datax')

    # test setter (not None)
    t.data1 = 2
    t.data2 = 2
    t.data3 = 2
    t.datax = 2

    # test that get returns it
    assert t.data1 == 2
    assert t.data2 == 2
    assert t.data3 == 2
    assert t.datax == 2

    # test that the private attribute saves the correct value
    assert t._data1 == 2
    assert t._data2 == 2
    assert t._data3 == 2
    assert t._wrong == 2
    assert not hasattr(t, '_datax')

    # test deleter
    del t.data1
    del t.data2
    del t.data3
    del t.datax

    # test that the private attribute is the default again
    assert t._data1 is None
    assert t._data2 is None
    assert t._data3 is None
    assert t._wrong is None
    assert not hasattr(t, '_datax')

    # test the getter returns it
    # Important test getter after the private acces because the getter would
    # set it if there were no private.
    assert t.data1 is None
    assert t.data2 is None
    assert t.data3 is None
    assert t.datax is None

    # test setter (None) but first set them again
    t.data1 = 2
    t.data2 = 2
    t.data3 = 2
    t.datax = 2

    t.data1 = None
    t.data2 = None
    t.data3 = None
    t.datax = None

    # test that the private is None
    assert t._data1 is None
    assert t._data2 is None
    assert t._data3 is None
    assert t._wrong is None
    assert not hasattr(t, '_datax')

    # Test that the getter returns the default
    assert t.data1 is None
    assert t.data2 is None
    assert t.data3 is None
    assert t.datax is None

    # test none of them copies by altering a list
    data = [1, 2, 3]
    t.data1 = data
    t.data2 = data
    t.data3 = data
    t.datax = data

    data[0] = 100

    # Test that the getter and private are altered as well
    assert t.data1[0] == 100
    assert t.data2[0] == 1    # the one who copied
    assert t.data3[0] == 100
    assert t.datax[0] == 100

    assert t._data1[0] == 100
    assert t._data2[0] == 1   # the one who copied
    assert t._data3[0] == 100
    assert t._wrong[0] == 100
    assert not hasattr(t, '_datax')


def test_subclassing_advanced1():
    class AdvancedDescriptorDefault(AdvancedDescriptor):
        def create_default(self):
            return [1, 2, 3]

    class AdvancedDescriptorSubclassTest1(object):
        data1 = AdvancedDescriptorDefault('data1', copy=True)
        data2 = AdvancedDescriptorDefault('data2', copy=False)

    t = AdvancedDescriptorSubclassTest1()

    # Getter sets the private and returns the default
    assert t.data1 == [1, 2, 3]
    assert t.data2 == [1, 2, 3]

    # Test that changes propagate back
    data = t.data1
    data[0] = 20
    assert t.data1[0] == 20

    data = t.data2
    data[0] = 20
    assert t.data2[0] == 20

    # Delete the values again
    del t.data1
    del t.data2

    # Test that changes propagate back 2 (when creating the default)
    data = t.data1
    data[0] = 20
    assert t.data1[0] == 20

    data = t.data2
    data[0] = 20
    assert t.data2[0] == 20

    # Delete the values again
    del t.data1
    del t.data2

    # Test that the default is set, when set with None
    t.data1 = None
    t.data2 = None
    assert t._data1 == [1, 2, 3]
    assert t._data2 == [1, 2, 3]
    assert t.data1 == [1, 2, 3]
    assert t.data2 == [1, 2, 3]

    # Test that the default is set, when deleted
    del t.data1
    del t.data2
    assert t._data1 == [1, 2, 3]
    assert t._data2 == [1, 2, 3]
    assert t.data1 == [1, 2, 3]
    assert t.data2 == [1, 2, 3]


def test_subclassing_advanced2():

    class AdvancedDescriptorTest(AdvancedDescriptor):
        def process_value(self, instance, value):
            if value is None:
                raise ValueError()
            elif isinstance(value, list):
                raise TypeError()
            return value

    class AdvancedDescriptorConvert(AdvancedDescriptor):
        def process_value(self, instance, value):
            return list(value)

    class AdvancedDescriptorSubclassTest1(object):
        data1 = AdvancedDescriptorTest('data1', copy=True)
        data2 = AdvancedDescriptorTest('data2', copy=False)

    t = AdvancedDescriptorSubclassTest1()

    # We check the setter because everything else should already be tested.

    # First setting with None should not trigger the ValueError because None
    # will create a default and should NEVER enter the process_value method
    t.data1 = None
    t.data2 = None

    # Test that list raise the TypeError
    with pytest.raises(TypeError):
        t.data1 = [1, 2, 3]
    with pytest.raises(TypeError):
        t.data2 = [1, 2, 3]

    # Test that something other than a list or None passes:
    t.data1 = 2
    t.data2 = 2
    assert t.data1 == 2
    assert t.data2 == 2
    assert t._data1 == 2
    assert t._data2 == 2

    # Test that data1 copies and data2 not
    adict = {'a': 10}
    t.data1 = adict
    t.data2 = adict
    adict['a'] = 20
    assert t.data1['a'] == 10
    assert t.data2['a'] == 20

    # The case where it doesn't copy because process_value altered the value
    # cannot be tested here.


def test_subclassing_advanced3():

    class AdvancedDescriptorConvert(AdvancedDescriptor):
        def process_value(self, instance, value):
            return list(value)

    class AdvancedDescriptorSubclassTest1(object):
        data1 = AdvancedDescriptorConvert('data1', copy=True)
        data2 = AdvancedDescriptorConvert('data2', copy=False)

    t = AdvancedDescriptorSubclassTest1()

    # We check the setter because everything else should already be tested.

    # First setting with None should not trigger the ValueError because None
    # will create a default and should NEVER enter the process_value method
    t.data1 = None
    t.data2 = None

    # Test that not-iterables (for example numbers) raise a TypeError
    with pytest.raises(TypeError):
        t.data1 = 2
    with pytest.raises(TypeError):
        t.data2 = 2

    # Test that something iterable is converted to a list
    t.data1 = {'a'}
    t.data2 = {'a'}
    assert t.data1 == ['a']
    assert t.data2 == ['a']

    # Test that both copy
    alist = [1, 2, 3]
    t.data1 = alist
    t.data2 = alist
    alist[0] = 20
    assert t.data1[0] == 1
    assert t.data2[0] == 1

    # Now test that even though data1 has copy it is not deepcopied because
    # the process function only casts to list

    # This is not a Bug, this is a test that normally the values are not copied
    # TWICE. (which would be inefficient)
    alist = [[1, 1, 1], 2, 2]
    t.data1 = alist
    t.data2 = alist
    alist[0][0] = 20
    assert t.data1[0][0] == 20
    assert t.data2[0][0] == 20


# No need to test wcs, flags or mask descriptor because they are simply the
# base descriptor...


def test_meta_descriptor():

    class MetaDescriptorTest(object):
        # descriptor without doc and copy=True
        meta1 = descriptors.Meta('meta1', doc='test', copy=True)
        meta2 = descriptors.Meta('meta2', doc='test', copy=False)

    assert MetaDescriptorTest.meta1.__doc__ == 'test'
    assert MetaDescriptorTest.meta2.__doc__ == 'test'

    t = MetaDescriptorTest()

    # Check default
    assert isinstance(t.meta1, collections.OrderedDict)
    assert isinstance(t.meta2, collections.OrderedDict)

    # Set to valid meta-objects
    for i in ({},
              collections.defaultdict(int),
              collections.OrderedDict(),
              collections.Counter(),
              fits.Header()):
        t.meta1 = i
        t.meta2 = i
        assert t.meta1.__class__ == i.__class__
        assert t.meta2.__class__ == i.__class__

        # test copy
        i['a'] = 10
        assert 'a' not in t.meta1
        assert t.meta2['a'] == 10

    # Make sure it doesn't let not-Mappings pass, for example integer
    with pytest.raises(TypeError):
        t.meta1 = 10
    with pytest.raises(TypeError):
        t.meta2 = 10


def test_data_descriptor():

    class DataDescriptorTest(object):
        data1 = descriptors.Data('data1', copy=False)
        data2 = descriptors.Data('data2', copy=True)

    t = DataDescriptorTest()

    # Has no default
    assert t.data1 is None
    assert t.data2 is None

    # Valid inputs are: numbers, lists, nested lists, numpy arrays
    # other stuff that looks like a numpy array (don't test the last one...)
    for i in (1, 1.5, 2+4j,
              [1, 2, 3], [[1, 1], [1, 1], [1, 1]],
              np.arange(100), np.ones((5, 5, 5, 5, 5))):
        t.data1 = i
        t.data2 = i
        assert isinstance(t.data1, np.ndarray)
        assert isinstance(t.data2, np.ndarray)

    # Numbers but not convertable:
    for i in (Fraction(1, 4), Decimal('3.14')):
        with pytest.raises(TypeError):
            t.data1 = i
        with pytest.raises(TypeError):
            t.data2 = i

    # Test copy: numpy array
    data = np.ones((3, 3))
    t.data1 = data
    t.data2 = data
    data[0, 0] = 50
    assert t.data1[0, 0] == 50
    assert t.data2[0, 0] == 1

    # Now a special case: Nested lists make sure they are deepcopied even if
    # they were changed in process_value
    data = [[1, 1], [2, 2]]
    t.data1 = data
    t.data2 = data
    data[0][0] = 100
    assert t.data1[0, 0] == 1
    assert t.data2[0, 0] == 1


def test_unit_descriptor():

    class UnitDescriptorTest(object):
        unit1 = descriptors.Unit('unit1', copy=False)
        unit2 = descriptors.Unit('unit2', copy=True)

    t = UnitDescriptorTest()

    # Test allowed cases: unit or string
    for i in (u.m, u.m*u.s/u.kg,
              'm', 'm*s/kg'):

        t.unit1 = i
        t.unit2 = i
        assert isinstance(t.unit1, (u.UnitBase, u.Unit))
        assert isinstance(t.unit2, (u.UnitBase, u.Unit))

        # no need to test copy since we always create a new unit.


def test_uncertainty_descriptor():

    class UncertaintyDescriptorTest(object):
        uncert1 = descriptors.Uncertainty('uncert1', copy=False)
        uncert2 = descriptors.Uncertainty('uncert2', copy=True)

    t = UncertaintyDescriptorTest()

    from ...nddata.meta import NDUncertainty
    from ...nddata import UnknownUncertainty, StdDevUncertainty

    # Test not-NDUncertainty will be wrapped in unknown uncertainty
    data = np.ones((3, 3))
    t.uncert1 = data
    t.uncert2 = data

    assert isinstance(t.uncert1, UnknownUncertainty)
    assert isinstance(t.uncert2, UnknownUncertainty)
    assert t.uncert1.parent_nddata is t
    assert t.uncert2.parent_nddata is t

    # Test if it copies it
    data[0, 0] = 2
    assert t.uncert1.array[0, 0] == 2
    assert t.uncert2.array[0, 0] == 1

    # Test unknownuncertainty
    data = np.ones((3, 3))
    t.uncert1 = UnknownUncertainty(data, copy=False)
    t.uncert2 = UnknownUncertainty(data, copy=False)

    assert isinstance(t.uncert1, UnknownUncertainty)
    assert isinstance(t.uncert2, UnknownUncertainty)
    assert t.uncert1.parent_nddata is t
    assert t.uncert2.parent_nddata is t

    # Test if it copies it - that was the reason I didn't copy while creating
    # the UnknownUncertainty
    data[0, 0] = 2
    assert t.uncert1.array[0, 0] == 2
    assert t.uncert2.array[0, 0] == 1

    # Test stddevuncertainty
    data = np.ones((3, 3))
    t.uncert1 = StdDevUncertainty(data, copy=False)
    t.uncert2 = StdDevUncertainty(data, copy=False)

    assert isinstance(t.uncert1, StdDevUncertainty)
    assert isinstance(t.uncert2, StdDevUncertainty)
    assert t.uncert1.parent_nddata is t
    assert t.uncert2.parent_nddata is t

    # Test if it copies it - that was the reason I didn't copy while creating
    # the StdDevUncertainty
    data[0, 0] = 2
    assert t.uncert1.array[0, 0] == 2
    assert t.uncert2.array[0, 0] == 1

    # Check that a new instance is created when assigning the uncertainty from
    # another instance
    another = UncertaintyDescriptorTest()
    another.uncert1 = t.uncert1
    another.uncert2 = t.uncert2
    # Make sure every uncertainty still links to it's parent
    assert t.uncert1.parent_nddata is t
    assert t.uncert2.parent_nddata is t
    assert another.uncert1.parent_nddata is another
    assert another.uncert2.parent_nddata is another
    # Check that they are different objects but with the same class
    assert t.uncert1 is not another.uncert1
    assert t.uncert2 is not another.uncert2
    assert t.uncert1.__class__ == another.uncert1.__class__
    assert t.uncert2.__class__ == another.uncert2.__class__

    # Change the originals in order to check if the array is shared:
    t.uncert1.array[1, 1] = 100
    t.uncert2.array[1, 1] = 100
    assert another.uncert1.array[1, 1] == 100
    assert another.uncert2.array[1, 1] == 1
