"""
Unit test for lvm_read.py
"""

import os
import sys

import numpy as np
import pytest

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')

from pyTrigger import pyTrigger


def test_trigger_at_first_data():
    pt = pyTrigger(rows=5, channels=2, trigger_channel=0, trigger_level=1, presamples=2)
    assert pt.triggered == False
    assert pt.rows_left == 5
    assert pt.finished == False
    assert pt.add_data(np.arange(4).reshape(2, 2)) == False
    assert pt.triggered == True
    assert pt.rows_left == 2
    assert pt.finished == False
    assert pt.add_data(-np.arange(4).reshape(2, 2)) == True
    assert pt.triggered == True
    assert pt.rows_left == 0
    assert pt.finished == True
    with pytest.raises(Exception):
        pt.add_data(-np.arange(4).reshape(2, 2))
    data = np.array([[0., 0.],
                     [0., 1.],
                     [2., 3.],
                     [0., -1.],
                     [-2., -3.]])
    np.testing.assert_array_equal(data, pt.get_data())


def test_trigger_at_first_data_one_chunk():
    pt = pyTrigger(rows=5, channels=2, trigger_channel=0, trigger_level=1, presamples=2)
    assert pt.triggered == False
    assert pt.rows_left == 5
    assert pt.finished == False
    pt.add_data(np.arange(8).reshape(4, 2))
    assert pt.triggered == True
    assert pt.rows_left == 0
    assert pt.finished == True
    data = np.array([[0., 0.],
                     [0., 1.],
                     [2., 3.],
                     [4., 5.],
                     [6., 7.]])
    np.testing.assert_array_equal(data, pt.get_data())


def test_trigger_up():
    pt = pyTrigger(rows=5, channels=2, trigger_channel=0, trigger_level=1,
                   trigger_type='up', presamples=2)
    assert pt.triggered == False
    assert pt.rows_left == 5
    assert pt.finished == False
    assert pt.add_data(-np.arange(4).reshape(2, 2)) == False
    assert pt.triggered == False
    assert pt.rows_left == 5
    assert pt.finished == False
    assert pt.add_data(-np.arange(4).reshape(2, 2)) == False
    assert pt.triggered == False
    assert pt.rows_left == 5
    assert pt.finished == False
    assert pt.add_data(np.arange(4).reshape(2, 2)) == False
    assert pt.triggered == True
    assert pt.rows_left == 2
    assert pt.finished == False
    assert pt.add_data(np.arange(4).reshape(2, 2)) == True
    assert pt.triggered == True
    assert pt.rows_left == 0
    assert pt.finished == True
    data = np.array([[-2., -3.],
                     [0., 1.],
                     [2., 3.],
                     [0., 1.],
                     [2., 3.]])
    np.testing.assert_array_equal(data, pt.get_data())


def test_trigger_down():
    pt = pyTrigger(rows=5, channels=2, trigger_channel=0, trigger_level=-1,
                   trigger_type='down', presamples=2)
    assert pt.triggered == False
    assert pt.rows_left == 5
    assert pt.finished == False
    assert pt.add_data(np.arange(4).reshape(2, 2)) == False
    assert pt.triggered == False
    assert pt.rows_left == 5
    assert pt.finished == False
    assert pt.add_data(np.arange(4).reshape(2, 2)) == False
    assert pt.triggered == False
    assert pt.rows_left == 5
    assert pt.finished == False
    assert pt.add_data(-np.arange(4).reshape(2, 2)) == False
    assert pt.triggered == True
    assert pt.rows_left == 2
    assert pt.finished == False
    assert pt.add_data(-np.arange(4).reshape(2, 2)) == True
    assert pt.triggered == True
    assert pt.rows_left == 0
    assert pt.finished == True
    data = np.array([[2., 3.],
                     [0., -1.],
                     [-2., -3.],
                     [0., -1.],
                     [-2., -3.]])
    np.testing.assert_array_equal(data, pt.get_data())


def test_trigger_abs():
    pt = pyTrigger(rows=5, channels=2, trigger_channel=0, trigger_level=1,
                   trigger_type='abs', presamples=2)
    assert pt.triggered == False
    assert pt.rows_left == 5
    assert pt.finished == False
    assert pt.add_data(0 * np.arange(4).reshape(2, 2)) == False
    assert pt.triggered == False
    assert pt.rows_left == 5
    assert pt.finished == False
    assert pt.add_data(-np.arange(4).reshape(2, 2)) == False
    assert pt.triggered == True
    assert pt.rows_left == 2
    assert pt.finished == False
    assert pt.add_data(-np.arange(4).reshape(2, 2)) == True
    assert pt.triggered == True
    assert pt.rows_left == 0
    assert pt.finished == True
    data = np.array([[0., 0.],
                     [0., -1.],
                     [-2., -3.],
                     [0., -1.],
                     [-2., -3.]])
    np.testing.assert_array_equal(data, pt.get_data())


def test_add_long_data():
    pt = pyTrigger(rows=5, channels=2, trigger_channel=0, trigger_level=1, presamples=2)
    assert pt.triggered == False
    assert pt.rows_left == 5
    assert pt.finished == False
    #    with pytest.raises(Exception):
    pt.add_data(np.arange(12).reshape(6, 2))
    assert pt.triggered == True
    assert pt.rows_left == 0
    assert pt.finished == True
    data = np.array([[0., 0.],
                     [0., 1.],
                     [2., 3.],
                     [4., 5.],
                     [6., 7.]])
    np.testing.assert_array_equal(data, pt.get_data())


def test_add_long_data2():
    data = np.array([[0., 0.],
                     [0., 1.],
                     [2., 3.],
                     [4., 5.],
                     [6., 7.],
                     [2., 3.],
                     [4., 5.],
                     [6., 7.],
                     [4., 5.],
                     [6., 7.]])

    pt = pyTrigger(rows=6, channels=2, trigger_channel=0, trigger_level=5, presamples=2)
    pt.add_data(data)
    assert pt.triggered == True
    assert pt.rows_left == 0
    assert pt.finished == True
    np.testing.assert_array_equal(data[2:8], pt.get_data())


def test_add_long_data3():
    data = np.array([[0., 0.],
                     [0., 1.],
                     [2., 3.],
                     [4., 5.],
                     [-6., 7.],
                     [2., 3.],
                     [4., 5.],
                     [6., 7.],
                     [4., 5.],
                     [6., 7.]])

    pt = pyTrigger(rows=6, channels=2, trigger_channel=0, trigger_level=-5, presamples=2, trigger_type='down')
    pt.add_data(data)
    assert pt.triggered == True
    assert pt.rows_left == 0
    assert pt.finished == True
    np.testing.assert_array_equal(data[2:8], pt.get_data())


def test_add_long_data():
    up_to = 10
    data = np.arange(6*up_to).reshape((-1,2))
    test_data = np.arange(start=-4, stop=6*up_to).reshape((-1,2))
    test_data[:2,:] = np.zeros((2,2), dtype=float)

    # changing trigger level
    for i in range(up_to):
        pt = pyTrigger(rows=5, channels=2, trigger_channel=0, trigger_level=2*i-0.1, presamples=2)
        pt.add_data(data)
        assert pt.triggered == True
        assert pt.rows_left == 0
        assert pt.finished == True
        np.testing.assert_array_equal(test_data[i:i+5], pt.get_data())

    # changing data
    for i in range(up_to):
        pt = pyTrigger(rows=5, channels=2, trigger_channel=0, trigger_level=20, presamples=2)
        pt.add_data(data[i:])
        assert pt.triggered == True
        assert pt.rows_left == 0
        assert pt.finished == True
        np.testing.assert_array_equal(test_data[11:11+5], pt.get_data())

if __name__ == '__mains__':
    np.testing.run_module_suite()
