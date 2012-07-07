from . import DiffeomorphismEstimator


class DiffeoDynamics():
    """ 
        This estimates a DDS, by running a DiffeomorphismEstimator
        for each command. 
        
    """
    def __init__(self, ratios, match_method):
        """
            match_method: 
            
               MATCH_CONTINUOUS     || y0 - y1 ||
               MATCH_BINARY            y0 * y1
                assuming y is binary (0,1)

        """
        self.ratios = ratios
        self.match_method = match_method
        self.commands2dynamics = {}
        self.ratios = ratios
        self.commands2label = {}
        self.commands2u = {}

    def update(self, commands_index, y0, y1, u, label=None):
        # Asssigns an index to each distinct command
        if not commands_index in self.commands2dynamics:
            # initialize the estimator
            self.commands2dynamics[commands_index] = \
                DiffeomorphismEstimator(self.ratios, self.match_method)
            self.commands2label[commands_index] = label
            self.commands2u[commands_index] = u

        # call the update method for the given command
        de = self.commands2dynamics[commands_index]
        de.update(y0, y1)

    def publish(self, pub):
        for ui, de in self.commands2dynamics.items():
            section = pub.section('u_%d:%s' % (ui, self.commands2label[ui]))
            de.publish(section)
