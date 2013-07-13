from .exp_switcher import ExpSwitcher
from astatsa.expectation import Expectation
from astatsa.utils.outer_product import outer
from boot_agents.misc_utils.tensors_display import (pub_svd_decomp,
    pub_eig_decomp)
from bootstrapping_olympics import UnsupportedSpec
from reprep.plot_utils.axes import y_axis_set_min
import numpy as np

__all__ = ['EstStatsTh']


class EstStatsTh(ExpSwitcher):
    ''' 

    '''

    def init(self, boot_spec):
        ExpSwitcher.init(self, boot_spec)
        # TODO: check float
        if len(boot_spec.get_observations().shape()) != 1:
            raise UnsupportedSpec('I assume 2D signals.')

        self.yy = Expectation()
        self.ylogy = Expectation()
        self.einvy = Expectation()
        self.ey = Expectation()
        self.elogy = Expectation()

    def merge(self, other):
        self.yy.merge(other.yy)
        self.ylogy.merge(other.ylogy)
        self.einvy.merge(other.einvy)
        self.ey.merge(other.ey)
        self.elogy.merge(other.elogy)
        
    def process_observations(self, obs):
        y = obs['observations']
        dt = obs['dt'].item()
        z = y == 0
        y[z] = 0.5
        yy = outer(y, y)
        dt = 1
        ylogy = outer(np.log(y), y)
        
        self.yy.update(yy, dt)
        self.ylogy.update(ylogy, dt)

        invy = 1.0 / y
        self.einvy.update(invy, dt)
        self.ey.update(y, dt)
        self.elogy.update(np.log(y), dt)

    def publish(self, pub):
        yy = self.yy.get_value()
        ylogy = self.ylogy.get_value()
        
        def symmetrize(x):
            return 0.5 * (x + x.T)
        yy = symmetrize(yy)
        ylogy = symmetrize(ylogy)
        
        
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
            y_axis_set_min(pylab, 0)


def get_deriv_matrix(n):
    a = np.zeros((n, n))
    for i in range(n):
        if i > 0:
            a[i - 1, i] = +1
        if i < n - 1:
            a[i + 1, i] = -1
    return a
        
    
