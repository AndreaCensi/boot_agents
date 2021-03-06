from . import Queue
from contracts import contract 

__all__ = ['DerivativeBox', 'DerivativeBox2']


class DerivativeBox():
    """ Implements 3-tap approximation to compute the derivative. """
    
    # TODO: do not use dt
    def __init__(self):
        self.q_y = Queue(3)
        self.q_dt = Queue(3)

    @contract(y='array', dt='>=0')
    def update(self, y, dt):
        if dt == 0:
            return  # XXX
        self.q_y.update(y)
        self.q_dt.update(dt)

    def ready(self):
        return self.q_y.ready()

    def get_value(self):
        assert self.ready()
        y = self.q_y.get_list()
        dt = self.q_dt.get_list()
        tdiff = dt[1] + dt[2]
        delta = y[-1] - y[0]
        sync_y_dot = delta / tdiff
        sync_y = y[1]
        return sync_y, sync_y_dot

    def reset(self):
        self.q_y.reset()
        self.q_dt.reset()



class DerivativeBox2():
    """ Implements 2-tap approximation to compute the derivative. """
    
    def __init__(self):
        self.q_y = Queue(2)
        self.q_dt = Queue(2)

    @contract(y='array', dt='>=0')
    def update(self, y, dt):
        if dt == 0:
            return  # XXX
        self.q_y.update(y)
        self.q_dt.update(dt)

    def ready(self):
        return self.q_y.ready()

    def get_value(self):
        assert self.ready()
        y = self.q_y.get_list()
        dt = self.q_dt.get_list()
        tdiff = dt[1]
        delta = y[1] - y[0]
        sync_y_dot = delta / tdiff
        sync_y = y[1]
        return sync_y, sync_y_dot

    def reset(self):
        self.q_y.reset()
        self.q_dt.reset()


