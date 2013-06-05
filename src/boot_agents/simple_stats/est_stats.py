from .exp_switcher import ExpSwitcher
from boot_agents.utils import MeanCovariance, cov2corr
from bootstrapping_olympics import UnsupportedSpec
from reprep.plot_utils import style_ieee_fullcol_xy
import numpy as np

__all__ = ['EstStats']


class EstStats(ExpSwitcher):
    ''' 
        A simple agent that estimates various statistics 
        of the observations. 
    '''

    def init(self, boot_spec):
        ExpSwitcher.init(self, boot_spec)
        if len(boot_spec.get_observations().shape()) != 1:
            raise UnsupportedSpec('I assume 1D signals.')

        self.y_stats = MeanCovariance()

    def process_observations(self, obs):
        y = obs['observations']
        dt = obs['dt'].item()
        self.y_stats.update(y, dt)

    def get_state(self):
        return dict(y_stats=self.y_stats)

    def set_state(self, state):
        self.y_stats = state['y_stats']

    def publish(self, pub):
        if self.y_stats.get_num_samples() == 0:
            pub.text('warning', 'Too early to publish anything.')
            return
        Py = self.y_stats.get_covariance()
        Ry = self.y_stats.get_correlation()
        Py_inv = self.y_stats.get_information()
        Ey = self.y_stats.get_mean()
        y_max = self.y_stats.get_maximum()
        y_min = self.y_stats.get_minimum()

        Ry0 = Ry.copy()
        np.fill_diagonal(Ry0, np.NaN)
        Py0 = Py.copy()
        np.fill_diagonal(Py0, np.NaN)

        pub.text('stats', 'Num samples: %s' % self.y_stats.get_num_samples())

        with pub.plot('y_bounds') as pylab:
            style_ieee_fullcol_xy(pylab)
            pylab.plot(Ey, label='E(y)')
            pylab.plot(y_max, label='y_max')
            pylab.plot(y_min, label='y_min')
            pylab.legend()

        all_positive = (np.min(Ey) > 0
                        and np.min(y_max) > 0
                        and np.min(y_min) > 0)
        if all_positive:
            with pub.plot('y_stats_log') as pylab:
                style_ieee_fullcol_xy(pylab)
                pylab.semilogy(Ey, label='E(y)')
                pylab.semilogy(y_max, label='y_max')
                pylab.semilogy(y_min, label='y_min')
                pylab.legend()

        pub.array_as_image('Py', Py, caption='cov(y)')
        pub.array_as_image('Py0', Py0, caption='cov(y) - no diagonal')

        pub.array_as_image('Ry', Ry, caption='corr(y)')
        pub.array_as_image('Ry0', Ry0, caption='corr(y) - no diagonal')

        pub.array_as_image('Py_inv', Py_inv)
        pub.array_as_image('Py_inv_n', cov2corr(Py_inv))

        with pub.plot('Py_svd') as pylab:  # XXX: use spectrum
            style_ieee_fullcol_xy(pylab)
            _, s, _ = np.linalg.svd(Py)
            s /= s[0]
            pylab.semilogy(s, 'bx-')

        with pub.subsection('y_stats') as sub:
            self.y_stats.publish(sub)

