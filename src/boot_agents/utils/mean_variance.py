from . import Expectation, contract, np, Publisher

__all__ = ['MeanVariance']


# TODO: write tests for this

class MeanVariance:

    ''' Computes mean and variance of some stream. '''
    def __init__(self, max_window=None):
        self.Ex = Expectation(max_window)
        self.Edx2 = Expectation(max_window)
        self.num_samples = 0

    @contract(x='array', dt='float,>0')
    def update(self, x, dt=1.0):
        self.num_samples += dt
        self.Ex.update(x, dt)
        dx = x - self.Ex()
        dx2 = dx * dx
        self.Edx2.update(dx2, dt)

    def assert_some_data(self):
        if self.num_samples == 0:
            raise Exception('Never updated')

    def get_mean(self):
        return self.Ex()

    def get_var(self):
        return self.Edx2()

    def get_std_dev(self):
        return np.sqrt(self.get_var())

    @contract(pub=Publisher)
    def publish(self, pub):
        if self.num_samples == 0:
            pub.text('warning',
                     'Cannot publish anything as I was never updated.')
            return

        pub.text('stats', 'Num samples: %s' % self.num_samples)

        mean = self.Ex()
        S = self.get_std_dev()

        if mean.ndim == 1:
            with pub.plot('mean') as pylab:
                pylab.plot(mean, 'k.')

            with pub.plot('std_dev') as pylab:
                pylab.plot(S, 'k.')
                a = pylab.axis()
                m = 0.1 * (a[3] - a[2])
                pylab.axis((a[0], a[1], 0, a[3] + m))
        else:
            pub.text('warning', 'Not implemented for ndim > 1')



