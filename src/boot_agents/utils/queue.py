import numpy as np

__all__ = ['Queue']


class Queue():
    ''' keeps the last num elements '''
    def __init__(self, num):
        self.l = []
        self.num = num

    def ready(self):
        ''' True if the list contains num elements. '''
        assert len(self.l) <= self.num
        return len(self.l) == self.num

    def update(self, value):
        self.l.append(value)
        while len(self.l) > self.num:
            self.l.pop(0)

    def get_all(self):
        # returns array[num, len(value) ]
        return np.array(self.l)

    def reset(self):
        self.l = []

    def get_list(self):
        return self.l

