import numpy as np

from .bdse_servo_robust import BDSEServoRobust, get_censored_y_goal


__all__ = ['BDSEServoRobustL1']


class BDSEServoRobustL1(BDSEServoRobust):
    """ 
        This servo strategy ignores points equal to 0 or 1
        and minimizes a L1 measure. 
    """
    
    def get_descent_direction(self, observations, goal):
#         if np.linalg.norm(observations - goal) == 0:
#             print observations
#             print goal
#             raise Exception('woah')
        # print id(observations), id(goal)
        y2, goal2 = get_censored_y_goal(observations, goal)
        u = self.bdse_model.get_servo_descent_direction_L1(y2, goal2)
        # print('servorobustL1: %s' % str(u))        
        return u
    
    def get_distance(self, y1, y2):
        a, b = get_censored_y_goal(y1, y2)
        return np.abs(a - b).sum()
