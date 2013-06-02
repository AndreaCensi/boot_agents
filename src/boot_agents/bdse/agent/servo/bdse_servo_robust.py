import numpy as np
from .bdse_servo_descent import BDSEServoFromDescent

__all__ = ['BDSEServoRobust']


class BDSEServoRobust(BDSEServoFromDescent):
    """ This servo strategy ignores points equal to 0 or 1. """
    
    def get_valid(self, y):
        return np.logical_and(y > 0, y < 1)
        
    def get_censored_y_goal(self, y, goal):
        valid = np.logical_and(self.get_valid(y), self.get_valid(goal))
        invalid = np.logical_not(valid)
        
        y = self.y.copy()
        goal = self.goal.copy()
        y[invalid] = 0
        goal[invalid] = 0
        return y, goal
    
    def get_descent_direction(self, observations, goal):
        y, goal = self.get_censored_y_goal(observations, goal)
        return self.bdse_model.get_servo_descent_direction(y, goal)        

