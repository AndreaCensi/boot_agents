from . import contract

__all__ = ['RemoveDoubles']

class RemoveDoubles():
    ''' 
        Filters the stream, ignoring values that did not change enough,
        in the sense that a certain percentage of the measurements remained
        fixed. 
        
        This is useful for a stream that contains more than one sensor,
        and perhaps the one with the fastest frequency has the fewest sensels.
    '''
        
    @contract(fraction='>=0,<=1') 
    def __init__(self, fraction):
        ''' :param fraction: minimum fraction of readings that must have changed. '''
        self.fraction = fraction
        self.reset()
        
    def ready(self):
        ''' True if the list contains num elements. '''
        return self.is_ready
    
    @contract(value='array')
    def update(self, value):
        if self.last is None:
            self.last = value.copy()
            self.is_ready = True
        else:
            if value.shape != self.last.shape:
                msg = 'Shape changed (%s to %s)' % (self.last.shape, value.shape)
                raise ValueError(msg)
            num_changed = (value != self.last).sum()
            if num_changed > self.fraction * value.size:
                self.last = value.copy()
                self.is_ready = True
            else:
                self.is_ready = False
            self.last_num_changed = num_changed
#            print('Num changed: %d/%d fraction: %s ready: %s' % 
#                  (num_changed, value.size, self.fraction, self.is_ready))
            
    def get_value(self):
        assert self.is_ready
        return self.last
    
    def reset(self):
        self.is_ready = False
        self.last = None
        self.last_num_changed = None
     
