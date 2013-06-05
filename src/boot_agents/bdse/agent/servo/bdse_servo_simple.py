from .bdse_servo_descent import BDSEServoFromDescent
import warnings

__all__ = ['BDSEServoSimple']


class BDSEServoSimple(BDSEServoFromDescent):
    """ This is the simplest thing that works. """
    
    def get_descent_direction(self, observations, goal):
        u = self.bdse_model.get_servo_descent_direction(observations, goal)
        u[2] *= 0.25
        warnings.warn('remove')
        return u

