# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import inspect

from ..decorators import format_doc
from astropy.tests.helper import pytest


def test_format_doc_stringInput_simple():
    # Simple tests with string input

    docstring_fail = ''

    # Raises an valueerror if input is empty
    with pytest.raises(ValueError):
        @format_doc(docstring_fail)
        def testfunc_fail():
            pass

    docstring = 'test'

    # A first test that replaces an empty docstring
    @format_doc(docstring)
    def testfunc_1():
        pass
    assert inspect.getdoc(testfunc_1) == docstring

    # Test that it replaces an existing docstring
    @format_doc(docstring)
    def testfunc_2():
        '''not test'''
        pass
    assert inspect.getdoc(testfunc_2) == docstring


def test_format_doc_stringInput_format():
    # Tests with string input and formatting

    docstring = 'yes {0} no {opt}'

    # Raises an indexerror if not given the formatted args and kwargs
    with pytest.raises(IndexError):
        @format_doc(docstring)
        def testfunc1():
            pass

    # Test that the formatting is done right
    @format_doc(docstring, '/', opt='= life')
    def testfunc2():
        pass
    assert inspect.getdoc(testfunc2) == 'yes / no = life'

    # Test that we can include the original docstring

    docstring2 = 'yes {0} no {__doc__}'

    @format_doc(docstring2, '/')
    def testfunc3():
        '''= 2 / 2 * life'''
        pass
    assert inspect.getdoc(testfunc3) == 'yes / no = 2 / 2 * life'


def test_format_doc_objectInput_simple():
    # Simple tests with object input

    def docstring_fail():
        pass

    # Self input while the function has no docstring raises an error
    with pytest.raises(ValueError):
        @format_doc(docstring_fail)
        def testfunc_fail():
            pass

    def docstring0():
        '''test'''
        pass

    # A first test that replaces an empty docstring
    @format_doc(docstring0)
    def testfunc_1():
        pass
    assert inspect.getdoc(testfunc_1) == inspect.getdoc(docstring0)

    # Test that it replaces an existing docstring
    @format_doc(docstring0)
    def testfunc_2():
        '''not test'''
        pass
    assert inspect.getdoc(testfunc_2) == inspect.getdoc(docstring0)


def test_format_doc_objectInput_format():
    # Tests with object input and formatting

    def docstring():
        '''test {0} test {opt}'''
        pass

    # Raises an indexerror if not given the formatted args and kwargs
    with pytest.raises(IndexError):
        @format_doc(docstring)
        def testfunc_fail():
            pass

    # Test that the formatting is done right
    @format_doc(docstring, '+', opt='= 2 * test')
    def testfunc2():
        pass
    assert inspect.getdoc(testfunc2) == 'test + test = 2 * test'

    # Test that we can include the original docstring

    def docstring2():
        '''test {0} test {__doc__}'''
        pass

    @format_doc(docstring2, '+')
    def testfunc3():
        '''= 4 / 2 * test'''
        pass
    assert inspect.getdoc(testfunc3) == 'test + test = 4 / 2 * test'


def test_format_doc_selfInput_simple():
    # Simple tests with self input

    # Self input while the function has no docstring raises an error
    with pytest.raises(ValueError):
        @format_doc(None)
        def testfunc_fail():
            pass

    # Test that it keeps an existing docstring
    @format_doc(None)
    def testfunc_1():
        '''not test'''
        pass
    assert inspect.getdoc(testfunc_1) == 'not test'


def test_format_doc_selfInput_format():
    # Tests with string input which is '__doc__' (special case) and formatting

    # Raises an indexerror if not given the formatted args and kwargs
    with pytest.raises(IndexError):
        @format_doc(None)
        def testfunc_fail():
            '''dum {0} dum {opt}'''
            pass

    # Test that the formatting is done right
    @format_doc(None, 'di', opt='da dum')
    def testfunc1():
        '''dum {0} dum {opt}'''
        pass
    assert inspect.getdoc(testfunc1) == 'dum di dum da dum'

    # Test that we cannot recursivly insert the original documentation

    @format_doc(None, 'di')
    def testfunc2():
        '''dum {0} dum {__doc__}'''
        pass
    assert inspect.getdoc(testfunc2) == 'dum di dum '


def test_format_doc_onMethod():
    # Check if the decorator works on methods too, to spice it up we try double
    # decorator
    docstring = 'what we do {__doc__}'

    class TestClass(object):
        @format_doc(docstring)
        @format_doc(None, 'strange.')
        def test_method(self):
            '''is {0}'''
            pass

    assert inspect.getdoc(TestClass.test_method) == 'what we do is strange.'


# @pytest.mark.skipif('six.PY2')
def test_format_doc_onClass():
    # Check if the decorator works on classes too
    docstring = 'what we do {__doc__} {0}{opt}'

    @format_doc(docstring, 'strange', opt='.')
    class TestClass(object):
        '''is'''
        pass

    assert inspect.getdoc(TestClass) == 'what we do is strange.'
