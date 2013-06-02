from .bdse_servo_robust import BDSEServoRobust

__all__ = ['BDSEServoRobustL1']


class BDSEServoRobustL1(BDSEServoRobust):
    """ 
        This servo strategy ignores points equal to 0 or 1
        and minimizes a L1 measure. 
    """
    
    def get_descent_direction(self, observations, goal):
        y, goal = self.get_censored_y_goal(observations, goal)
        u = self.bdse_model.get_servo_descent_direction_L1(y, goal)        
        return u
    
