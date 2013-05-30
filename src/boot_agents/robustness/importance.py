from boot_agents.utils import MeanCovariance, generalized_gradient
from contracts import contract
from reprep.plot_utils import (turn_off_bottom_and_top, x_axis_set,
    y_axis_balanced, y_axis_set)
import numpy as np
import warnings
 
__all__ = ['Importance']

class Importance(object):
    
    def __init__(self, max_y_dot, max_gy, min_y, max_y):
        self.min_y = min_y
        self.max_y = max_y
        self.max_y_dot = max_y_dot
        self.max_gy = max_gy
        
        self.w_stats = MeanCovariance()
        
        warnings.warn('This is not invariant to linear transformation (max,min y).')
    
        self.once = False
        
    @contract(y='array[N]', y_dot='array[N]', returns='array[N]')
    def get_importance(self, y, y_dot):
        self.once = True        
    
        gy = generalized_gradient(y)  # gy='array[1xN]', 

        y_valid = np.logical_and(y > self.min_y, y < self.max_y)
        gy0 = gy[0, :] 
        gy_valid = np.abs(gy0) < self.max_gy
        y_dot_valid = np.abs(y_dot) < self.max_y_dot
        
        
        w = y_valid * 1.0 * gy_valid * y_dot_valid
        
        
        self.w_stats.update(w)
        self.last_w = w  
        self.last_y = y
        self.last_y_valid = y_valid
        self.last_y_dot = y_dot
        self.last_y_dot_valid = y_dot_valid
        self.last_gy = gy
        self.last_gy_valid = gy_valid

        return w
    
    def publish(self, pub):
        if not self.once:
            pub.text('info', 'never called yet')
            return
        
        N = self.last_y.size
        
        with pub.plot('last_w') as pylab:
            pylab.plot(self.last_w, 's')
            
            x_axis_set(pylab, -1, N)
            y_axis_set(pylab, -0.1, 1.1)
            turn_off_bottom_and_top(pylab)

        gy0 = self.last_gy[0, :]

        def plot_good_bad(pylab, x, valid):
            invalid = np.logical_not(valid)
            pylab.plot(np.nonzero(valid)[0], x[valid], 'ks')
            pylab.plot(np.nonzero(invalid)[0], x[invalid], 'rs')
            

        with pub.plot('last_y') as pylab: 
            plot_good_bad(pylab, self.last_y, self.last_y_valid)
            y_axis_set(pylab, -0.1, +1.1)
            x_axis_set(pylab, -1, N)
            turn_off_bottom_and_top(pylab)
            
        with pub.plot('last_y_dot') as pylab:
            pylab.plot(self.last_y_dot)            
            plot_good_bad(pylab, self.last_y_dot, self.last_y_dot_valid)
            
            upper = np.ones(N) * self.max_y_dot
            lower = -upper
            pylab.plot(upper, 'r--')
            pylab.plot(lower, 'r--')
            
            x_axis_set(pylab, -1, N)
            y_axis_balanced(pylab)
            turn_off_bottom_and_top(pylab)

        with pub.plot('last_gy') as pylab:            
            plot_good_bad(pylab, gy0, self.last_gy_valid)
            
            upper = np.ones(N) * self.max_gy
            lower = -upper
            pylab.plot(upper, 'r--')
            pylab.plot(lower, 'r--')
            
            x_axis_set(pylab, -1, N)
            y_axis_balanced(pylab)
            turn_off_bottom_and_top(pylab)

 
        self.w_stats.publish(pub.section('w_stats'))
