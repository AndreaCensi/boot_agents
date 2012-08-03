from bootstrapping_olympics import (BootSpec, RobotInterface, RobotObservations,
    EpisodeDesc)
from contracts import contract


class BDSSystem(RobotInterface):
    
    def __init__(self, y0_generator, diff_model, dt):
        self.y0_generator = y0_generator
        self.diff_model = diff_model
        self.dt = dt
        
    @contract(returns=BootSpec)
    def get_spec(self):
        ''' Returns the sensorimotor spec for this robot
            (a BootSpec object). '''

    @contract(returns=EpisodeDesc)
    def new_episode(self):
        ''' 
            Skips to the next episode. 
            In real robots, the platform might return to start position.
            
            Guaranteed to be called at least once before get_observations().
            
            Should return an instance of EpisodeDesc.
        '''

    @contract(commands='array', commands_source='str')
    def set_commands(self, commands, commands_source):
        ''' Send the given commands. '''

    @contract(returns=RobotObservations)
    def get_observations(self):
        ''' Get observations. Must return an instance of RobotObservations. '''

