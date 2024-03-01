# -*- coding: UTF-8 -*-
import numpy as np

__version__ = '0.12.2'

class RingBuffer2D:
    """A 2D ring buffer using numpy arrays

    Class for 2D buffer array.
    Based on this code: http://scimusing.wordpress.com/2013/10/25/ring-buffers-in-pythonnumpy/
    """

    def __init__(self, rows, columns, dtype='float'):
        """
        Initialize ring buffer.

        :param rows: # of rows in the 2D buffer
        :param columns: # of columns in the 2D buffer
        :param dtype: default 'float'
        """
        self.rows = rows
        self.columns = columns
        self.data = np.zeros((self.rows, self.columns), dtype=dtype)
        self.index = 0

    def clear(self):
        """Clear buffer."""
        self.data = np.zeros_like(self.data)
        self.index = 0

    def extend(self, data):
        """adds array `data` to ring buffer"""
        rows_to_add = len(data)
        if rows_to_add == 0 or len(data[0]) != self.columns:
            return
        rows_to_add_index = (self.index + np.arange(rows_to_add)) % self.rows
        self.data[rows_to_add_index] = data
        self.index = rows_to_add_index[-1] + 1

    def get_data(self):
        """Returns the first-in-first-out data in the ring buffer"""
        rows_index = (self.index + np.arange(self.rows)) % self.rows
        return self.data[rows_index]


class pyTrigger:
    """
    Software trigger with ring buffer.
    """

    def __init__(self,
                 rows=5120,
                 channels=4,
                 trigger_channel=0,
                 trigger_level=1.,
                 trigger_type='up',
                 presamples=1000):
        """
        Software trigger with ring buffer.

        :param rows: # of rows
        :param channels: # of channels
        :param trigger_channel: the channel used for triggering
        :param trigger_level: the level to cross, to start trigger
        :param trigger_type: 'up' is default, possible also 'down'/'abs'
        :param presamples: # of presamples
        """
        self.rows = rows
        self.channels = channels
        self.trigger_channel = trigger_channel
        self.trigger_level = trigger_level
        self.trigger_type = trigger_type.lower()
        self.presamples = presamples
        self.ringbuff = RingBuffer2D(rows=self.rows, columns=self.channels)
        self.triggered = False
        self.rows_left = self.rows
        self.finished = False
        self.first_data = True

    def _trigger_index(self, data):
        """Get the index where the trigger is found"""
        row = data[:, self.trigger_channel]
        if self.trigger_type == 'abs':
            trigger = np.argwhere(np.abs(row) > np.abs(self.trigger_level))
        elif self.trigger_type == 'down':
            trigger = np.argwhere(row < self.trigger_level)
        else:
            trigger = np.argwhere(row > self.trigger_level)
        if len(trigger) > 0:
            trigger = trigger[0][0]
            self.triggered = True
        return trigger

    def _add_data_to_buffer(self, data):
        """Add data to ring buffer

        :param data:
        :return: none
        """
        if self.rows_left > len(data):
            take_rows = len(data)
        else:
            take_rows = self.rows_left
        self.ringbuff.extend(data[:take_rows])
        if self.triggered:
            self.rows_left -= take_rows
            if self.rows_left == 0:
                self.finished = True
        self.first_data = False

    def add_data(self, data):
        """Add data and check for trigger

        Data is typically added until the return is `True`, meaning trigger was successful
        :param data: data to be added
        :return: finished: True if trigger finished
        """
        for i in range(len(data) // self.rows + 1):
            self._add_data_chunk(data[i * self.rows:(i + 1) * self.rows])
            if self.finished:
                return self.finished
        return self.finished

    def _add_data_chunk(self, data):
        if len(data) > self.rows:
            raise Exception('Length of data should not be longer than the number of rows.')
        if self.finished:
            raise Warning('Triggering finished.')
        if self.triggered:
            self._add_data_to_buffer(data)
            return
        trigger = self._trigger_index(data)
        if self.triggered:
            if trigger < self.presamples:
                if self.first_data:
                    prepend_zero_rows = self.presamples - trigger
                    self._add_data_to_buffer(np.zeros(shape=(prepend_zero_rows, self.channels), dtype='float'))
                else:
                    self.rows_left = self.rows_left - self.presamples + trigger
                self._add_data_to_buffer(data)
                return
            else:
                self.rows_left = self.rows_left + trigger - self.presamples
                self._add_data_to_buffer(data)
                return

        else:
            self._add_data_to_buffer(data)
            return

    def get_data(self):
        """
        Get the triggered data

        :return:
        """
        return self.ringbuff.get_data()

if __name__ == '__main__':
    up_to = 10
    data = np.arange(4 * up_to).reshape((-1, 2))
    test_data = np.arange(start=-4, stop=4 * up_to).reshape((-1, 2))
    test_data[:2, :] = np.zeros((2, 2), dtype=float)

    i = 0
    pt = pyTrigger(rows=5, channels=2, trigger_channel=0, trigger_level=-0.1, presamples=2)

    pt.add_data(data)
    print(i)
    assert pt.triggered == True
    assert pt.rows_left == 0
    assert pt.finished == True
    np.testing.assert_array_equal(test_data[i:i + 5], pt.get_data())
