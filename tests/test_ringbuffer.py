"""
Unit test for lvm_read.py
"""

import os
import sys

import numpy as np

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')

from pyTrigger import RingBuffer2D


def test_ringbuff_1():
    test_comparison_data = np.array([
        [[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 1., 2.],
         [3., 4., 5.]],
        [[0., 0., 0.],
         [0., 1., 2.],
         [3., 4., 5.],
         [6., 7., 8.],
         [9., 10., 11.]],
        [[3., 4., 5.],
         [6., 7., 8.],
         [9., 10., 11.],
         [12., 13., 14.],
         [15., 16., 17.]]
    ])

    samples = 5
    channels = 3
    ringbuff = RingBuffer2D(rows=samples, columns=channels)

    fill_rows_at_a_time = 2
    for i in range(3):
        data = i * (channels * fill_rows_at_a_time) + np.arange(channels * fill_rows_at_a_time, dtype='float')
        data = data.reshape((fill_rows_at_a_time, channels))
        ringbuff.extend(data)  # write
        np.testing.assert_array_almost_equal(test_comparison_data[i], ringbuff.get_data())


if __name__ == '__mains__':
    np.testing.run_module_suite()
