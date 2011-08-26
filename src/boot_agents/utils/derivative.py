from . import Queue

class DerivativeBox():
    
    def __init__(self):
        self.q_y = Queue(3)
        self.q_dt = Queue(3)
        
    def update(self, y, dt):
        ''' returns y, y_dot or None, None if the queue is not full '''
        assert dt > 0 
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

    
