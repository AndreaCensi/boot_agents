from astatsa.expectation import Expectation
from astatsa.utils import outer
from blocks.interface import Sink
from blocks.library.timed.checks import check_timed_named
from bootstrapping_olympics import LearningAgent, UnsupportedSpec
from reprep.plot_utils.axes import y_axis_set_min
import numpy as np
from bootstrapping_olympics.interfaces.agent import BasicAgent



__all__ = ['EstStatsTh']


class EstStatsTh(BasicAgent, LearningAgent):
    ''' 

    '''

    def init(self, boot_spec):
        # TODO: check float
        if len(boot_spec.get_observations().shape()) != 1:
            raise UnsupportedSpec('I assume 2D signals.')

        self.yy = Expectation()
        self.ylogy = Expectation()
        self.einvy = Expectation()
        self.ey = Expectation()
        self.elogy = Expectation()
    
        from boot_agents.utils.mean_covariance import MeanCovariance

        self.covy = MeanCovariance()
        self.covfy = MeanCovariance()
         
    def merge(self, other):
        self.yy.merge(other.yy)
        self.ylogy.merge(other.ylogy)
        self.einvy.merge(other.einvy)
        self.ey.merge(other.ey)
        self.elogy.merge(other.elogy)
        self.covy.merge(other.covy)
        self.covfy.merge(other.covfy)
        
        
    def get_learner_as_sink(self):
        class LearnSink(Sink):
            def __init__(self, agent):
                self.agent = agent
            def reset(self):
                pass
            def put(self, value, block=True, timeout=None):  # @UnusedVariable
                check_timed_named(value)
                timestamp, (signal, x) = value  # @UnusedVariable
                if not signal in ['observations', 'commands']:
                    msg = 'Invalid signal %r to learner.' % signal
                    raise ValueError(msg)
                
                if signal == 'observations':
                    self.agent.update(x)
                
        return LearnSink(self)
        
    def update(self, y, dt=1.0):
        n = y.size
#         # XXX
#         which = np.array(range(y.size)) < 100
#         y[which] = (y * y)[which]
        
        z = y == 0
        y[z] = 0.5
        yy = outer(y, y)
        dt = 1
        
        # TMP 
        logy = y * y 
#         logy[:int(n / 4)] = y[:int(n / 4)] 

        ylogy = outer(logy, y)
        
        self.yy.update(yy, dt)
        self.ylogy.update(ylogy, dt)

        invy = 1.0 / y
        self.einvy.update(invy, dt)
        self.ey.update(y, dt)
        self.elogy.update(logy, dt)

        self.covy.update(y, dt)
        self.covfy.update(logy, dt)

    def publish(self, pub):
#         pub.text('warn', 'using y^3')
        yy = self.yy.get_value()
        ylogy = self.ylogy.get_value()
        
        def symmetrize(x):
            return 0.5 * (x + x.T)
        yy = symmetrize(yy)
        ylogy = symmetrize(ylogy)
        
        from boot_agents.misc_utils import pub_eig_decomp, pub_svd_decomp
        from boot_agents.utils.mean_covariance import MeanCovariance

        with pub.subsection('yy') as sub:
            sub.array_as_image('val', yy)
            pub_svd_decomp(sub, yy)
            pub_eig_decomp(sub, yy)
            
        with pub.subsection('ylogy') as sub:
            sub.array_as_image('ylogy', ylogy)
            pub_svd_decomp(sub, ylogy)
            pub_eig_decomp(sub, ylogy)
            
        f = pub.figure()
        with f.plot('both') as pylab:
            pylab.plot(yy.flat, ylogy.flat, '.')
            pylab.xlabel('yy')
            pylab.ylabel('ylogy')
            pylab.axis('equal')
        
        D = get_deriv_matrix(yy.shape[0])
        inv = np.linalg.inv(yy)
        x = np.dot(ylogy, inv)
        
        f.array_as_image('inv', inv)
        f.array_as_image('ylogy_times_inverse', x)
        f.array_as_image('D', D)
        f.array_as_image('Dlog', np.dot(D, x))
        
        diag = np.diag(x)
        einvy = self.einvy.get_value()
        elogy = self.elogy.get_value()
        ey = self.ey.get_value()
        f = pub.figure()
        with f.plot('einvy', caption='E{1/y}') as pylab:
            pylab.plot(einvy, 's')
            y_axis_set_min(pylab, 0)
        with f.plot('elogy', caption='E{logy}') as pylab:
            pylab.plot(elogy, 's')
            # y_axis_set_min(pylab, 0)
        with f.plot('ey', caption='E{y}') as pylab:
            pylab.plot(ey, 's')
            y_axis_set_min(pylab, 0)
        with f.plot('diag', caption='diagonal of x') as pylab:
            pylab.plot(diag, 's')
            # y_axis_set_min(pylab, 0)


def get_deriv_matrix(n):
    a = np.zeros((n, n))
    for i in range(n):
        if i > 0:
            a[i - 1, i] = +1
        if i < n - 1:
            a[i + 1, i] = -1
    return a
        
    
