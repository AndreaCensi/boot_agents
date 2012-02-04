#Here are a couple references on computing sample variance.
#
#Chan, Tony F.; Golub, Gene H.; LeVeque, Randall J. (1983). 
#Algorithms for Computing the Sample Variance: Analysis and Recommendations. 
#The American Statistician 37, 242-247.
#
#Ling, Robert F. (1974). Comparison of Several Algorithms for Computing Sample 
#Means and Variances. Journal of the American Statistical Association,
# Vol. 69, No. 348, 859-866. 

from . import logger, Expectation, np, contract, Publisher, cov2corr, outer
from ..misc_utils.pylab_axis import y_axis_positive, y_axis_extra_space
from numpy.linalg.linalg import pinv, LinAlgError
from bootstrapping_olympics.utils import check_all_finite
from reprep.plot_utils.styles import style_ieee_fullcol_xy


class MeanCovariance:
    ''' Computes mean and covariance of a quantity '''
    def __init__(self, max_window=None):
        self.mean_accum = Expectation(max_window)
        self.covariance_accum = Expectation(max_window)
        self.minimum = None
        self.maximum = None
        self.num_samples = 0

    def get_num_samples(self):
        return self.num_samples

    def update(self, value, dt=1.0):
        check_all_finite(value)

        self.num_samples += dt

        n = value.size
        if  self.maximum is None:
            self.maximum = value.copy()
            self.minimum = value.copy()
            self.P_t = np.zeros(shape=(n, n), dtype=value.dtype)
        else:
            # TODO: check dimensions
            if not (value.shape == self.maximum.shape):
                raise ValueError('Value shape changed: %s -> %s' %
                                 (self.maximum.shape, value.shape))
            self.maximum = np.maximum(value, self.maximum)
            self.minimum = np.minimum(value, self.minimum)

        self.mean_accum.update(value, dt)
        mean = self.mean_accum.get_value()
        value_norm = value - mean

        check_all_finite(value_norm)
        P = outer(value_norm, value_norm)
        self.covariance_accum.update(P, dt)
        self.last_value = value

    def assert_some_data(self):
        if self.num_samples == 0:
            raise Exception('Never updated')

    def get_mean(self):
        self.assert_some_data()
        return self.mean_accum.get_value()

    def get_maximum(self):
        self.assert_some_data()
        return self.maximum

    def get_minimum(self):
        self.assert_some_data()
        return self.minimum

    def get_covariance(self):
        self.assert_some_data()
        return self.covariance_accum.get_value()

    def get_correlation(self):
        self.assert_some_data()
        corr = cov2corr(self.covariance_accum.get_value())
        np.fill_diagonal(corr, 1)
        return corr

    def get_information(self, rcond=1e-2):
        self.assert_some_data()
        try:
            P = self.get_covariance()
            return pinv(P, rcond=rcond)
        except LinAlgError:
            filename = 'pinv-failure'
            import pickle
            with  open(filename + '.pickle', 'w') as f:
                pickle.dump(self, f)
            logger.error('Did not converge; saved on %s' % filename)

    @contract(pub=Publisher, publish_information='bool')
    def publish(self, pub, publish_information=False):
        if self.num_samples == 0:
            pub.text('warning',
                     'Cannot publish anything as I was never updated.')
            return

        P = self.get_covariance()
        R = self.get_correlation()
        Ey = self.get_mean()
        y_max = self.get_maximum()
        y_min = self.get_minimum()

        pub.text('stats', 'Num samples: %s' % self.mean_accum.get_mass())

        with pub.plot('expectation') as pylab:
            style_ieee_fullcol_xy(pylab)
            pylab.plot(Ey, 's', label='expectation')
            pylab.plot(y_max, 's', label='max')
            pylab.plot(y_min, 's', label='min')
            y_axis_extra_space(pylab)
            pylab.legend()

        pub.array_as_image('covariance', P)
        R = R.copy()
        np.fill_diagonal(R, np.nan)
        pub.array_as_image('correlation', R)
        if publish_information:
            P_inv = self.get_information()
            pub.array_as_image('information', P_inv)

        with pub.plot('P_diagonal') as pylab:
            style_ieee_fullcol_xy(pylab)
            pylab.plot(P.diagonal(), 's')
            y_axis_positive(pylab)

        with pub.plot('P_diagonal_sqrt') as pylab:
            style_ieee_fullcol_xy(pylab)
            pylab.plot(np.sqrt(P.diagonal()), 's')
            y_axis_positive(pylab)


