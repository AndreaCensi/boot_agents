from . import DiffeomorphismEstimator

class DiffeoDynamics():
    
    def __init__(self, ratios, match_method):
        self.ratios = ratios
        self.match_method = match_method
        self.commands2dynamics = {}
        self.ratios = ratios
        self.commands2label = {}
        self.commands2u = {}
        
    def update(self, commands_index, y0, y1, u, label=None):
        if not commands_index in self.commands2dynamics:
            self.commands2dynamics[commands_index] = \
                DiffeomorphismEstimator(self.ratios, self.match_method)
            print('-initializing command %d (label: %s)' % (commands_index, label))
            self.commands2label[commands_index] = label
            self.commands2u[commands_index] = u
        de = self.commands2dynamics[commands_index]
        de.update(y0, y1) 
    
    def publish(self, pub):
        for ui, de in self.commands2dynamics.items():
            section = pub.section('u_%d:%s' % (ui, self.commands2label[ui]))
            de.publish(section) 
