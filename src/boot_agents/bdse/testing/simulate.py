
import numpy as np
from contracts import contract

__all__ = ['BDSSimulator']


class BDSSimulator(object):
    """ A simulator of BDS models """

    def __init__(self, model, y0_dist, u_dist):
        """
            y0 is a function returning the first state
            u_k returns the sequence of commands
        """
        self.model = model
        self.y0_dist = y0_dist
        self.u_dist = u_dist

    @contract(num_steps='int,>=1', T='float,>0')
    def get_simulation(self, num_steps, T):
        """ Returns a list of tuples containing (y, y_dot, u)
        
        """
        y_i = self.y0_dist()
        for _ in range(num_steps):
            u_i = self.u_dist()
            y_dot_i = self.model.get_y_dot(y_i, u_i)
            yield (np.array(y_i),
                   np.array(u_i),
                   np.array(y_dot_i))
            y_i = y_dot_i + T * y_dot_i
