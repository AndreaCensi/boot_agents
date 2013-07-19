from .exp_switcher import ExpSwitcher
from bootstrapping_olympics import UnsupportedSpec
from contracts import contract
import numpy as np
from boot_agents.utils.nonparametric import scale_score

__all__ = ['EstConditionalDistribution']


class EstConditional(object):
    """ 
        Estimates the joint distribution of two variables. 
        It is assumed that the two variables are normalized in [0, 1]. 
    """
    
    @contract(ncells='int,>=2')
    def __init__(self, ncells=100):
        self.mass = np.zeros((ncells, ncells))
        self.mass_total = 0
        self.disc = ncells
        
    @contract(x='float,>=0,<=1', y='float,>=0,<=1')
    def update(self, x, y, dt=1):
        ok_x = x > 0 and x < 1
        ok_y = y > 0 and y < 1
        if not (ok_x and ok_y):
            return
         
        u = self.cell_from_value(x)
        v = self.cell_from_value(y)
        self.mass[u, v] += dt
        self.mass_total += dt
        
    def cell_from_value(self, x):
        if x == 1:
            return self.disc - 1
        return np.floor(x * self.disc)

    def get_num_samples(self):
        return self.mass_total

    def get_probability(self):  # XXX
        return self.mass * (1.0 / self.mass_total)
    
    def get_conditional(self, x0):
        u = self.cell_from_value(x0)
        p = self.mass[u, :]
        ps = np.sum(p)
        if ps == 0:
            return 0 * p
        else:
            return p / ps
        
    def get_expected(self, x0):
        """ Expected value of b given a = x0 """
        p = self.get_conditional(x0)
        x = np.linspace(0, 1, p.size)
        return np.sum(p * x)
        
    def merge(self, other):
        assert other.disc == self.disc
        self.mass += other.mass
        self.mass_total += other.mass_total
        

class EstConditionalDistribution(ExpSwitcher):
    ''' 
        A simple agent that estimates various distributions of the 
        observations.
    '''

    @contract(index='int,>=0')
    def __init__(self, index, ncells=50, **kwargs):
        '''
        :param index: Which index of the observations to look at.
        '''
        self.index = index
        self.ncells = ncells
        ExpSwitcher.__init__(self, **kwargs)
        

    def init(self, boot_spec):
        ExpSwitcher.init(self, boot_spec)
        if len(boot_spec.get_observations().shape()) != 1:
            raise UnsupportedSpec('I assume 1D signals.')

        self.n = boot_spec.get_observations().size()
        if self.n <= self.index:
            msg = 'Too few observations (%s > %s)' % (self.index, self.n)
            raise UnsupportedSpec(msg)
            
        self.estimators = [EstConditional(self.ncells) for _ in range(self.n)]

    def merge(self, other):
        for i in range(self.n):
            self.estimators[i].merge(other.estimators[i])   
   
    def process_observations(self, obs):
        y = obs['observations']
        a = y[self.index]
        for i in range(self.n):
            self.estimators[i].update(a, y[i])
        
#     def get_state(self):
#         return dict(y_stats=self.y_stats)
# 
#     def set_state(self, state):
#         self.y_stats = state['y_stats']

    def publish(self, pub):
        if self.estimators[0].get_num_samples() == 0:
            pub.text('warning', 'Too early to publish anything.')
            return
        
        for i in range(self.n):
            with pub.subsection('c%03d' % i) as sub:
                f = sub.figure()
                sub.text('mass', self.estimators[i].get_num_samples())
                p = self.estimators[i].get_probability()
                p = scale_score(p)
                f.array_as_image('probability', p)
        
        alls = []         
        for a in np.linspace(0.05, 0.95, 10):
            with pub.subsection('a_%f' % a) as sub:
                sub.text('what', 'conditional for y = %s' % a)
                l = []
                for i in range(self.n): 
                    u = self.estimators[i].get_expected(a)
                    l.append(u)
               
                alls.append((a, l)) 
                f = sub.figure()
                with f.plot('expected') as pylab:
                    pylab.plot(l)
                    pylab.xlabel('i')
                    pylab.ylabel('E{yi|x=%s}' % a)
                    pylab.axis((-1, self.n, -0.05, +1.05))
                    
                    
        with pub.figure().plot('all') as pylab:
            for a, conds in alls:
                pylab.plot(conds, '-')
                pylab.plot(self.index, a, 'rs')
                
                    
            
            
            
            
