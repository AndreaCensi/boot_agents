import numpy as np
from .bdse_servo_descent import BDSEServoFromDescent
from boot_agents.robustness.importance import Importance

__all__ = ['BDSEServoRobust']


class BDSEServoRobust(BDSEServoFromDescent):
    """ This servo strategy ignores points equal to 0 or 1. """
    
    def get_descent_direction(self, observations, goal):
        y, goal = get_censored_y_goal(observations, goal)
        return self.bdse_model.get_servo_descent_direction(y, goal)        

    def get_distance(self, y1, y2):
        a, b = self.get_censored_y_goal(y1, y2)
        return np.abs(a - b).sum()

def get_valid(y):
    im = Importance(max_y_dot=1000, max_gy=0.01, min_y=0, max_y=1)
    y_dot = y * 0
    w = im.get_importance(y, y_dot)
    return w.astype('bool')
    # return np.logical_and(y > 0, y < 1)
    
def get_censored_y_goal(y, goal):
    print ''.join('%d' % int(x) for x in get_valid(y))
    valid = np.logical_and(get_valid(y), get_valid(goal))
    invalid = np.logical_not(valid)
    y = y.copy()
    goal = goal.copy()
    y[invalid] = 0
    goal[invalid] = 0
    return y, goal

