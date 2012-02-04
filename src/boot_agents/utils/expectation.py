from . import contract, np
from bootstrapping_olympics.utils.np_comparisons import check_all_finite


__all__ = ['Expectation', 'ExpectationSlow', 'ExpectationFast']


@contract(A='array', wA='>=0', B='array', wB='>=0')
def weighted_average(A, wA, B, wB):
    mA = wA / (wA + wB)
    mB = wB / (wA + wB)
    return (mA * A + mB * B)


class ExpectationSlow:
    ''' A class to compute the mean of a quantity over time '''
    def __init__(self, max_window=None):
        ''' 
            If max_window is given, the covariance is computed
            over a certain interval. 
        '''
        self.num_samples = 0
        self.value = None
        self.max_window = max_window

    @contract(value='array', dt='>0')
    def update(self, value, dt=1.0):
        if self.value is None:
            self.value = value
        else:
            self.value = weighted_average(self.value, float(self.num_samples),
                                          value, float(dt))
        self.num_samples += dt
        if self.max_window and self.num_samples > self.max_window:
            self.num_samples = self.max_window

    def get_value(self):
        return self.value

    def get_mass(self):
        return self.num_samples


class ExpectationFast:
    ''' A more efficient implementation. '''
    def __init__(self, max_window=None, extremely_fast=False):
        ''' 
            If max_window is given, the covariance is computed
            over a certain interval. 
            
            extremely_fast: saves memory; might crash on some python 
            interpreters
            TODO: put automatic tests to detect this
        '''
        self.max_window = max_window
        self.accum_mass = 0.0
        self.accum = None
        self.needs_normalization = True
        self.extremely_fast = extremely_fast

    @contract(cur_mass='float,>=0')
    def reset(self, cur_mass=1.0):
        self.accum = self.get_value()
        self.accum_mass = cur_mass

    @contract(value='array', dt='float,>=0')
    def update(self, value, dt=1.0):
        check_all_finite(value)

        if self.accum is None:
            self.accum = value * dt
            self.accum_mass = dt
            self.needs_normalization = True
            self.buf = np.empty_like(value)
            self.buf.fill(np.NaN)
            self.result = np.empty_like(value)
            self.result.fill(np.NaN)
        else:
            if self.extremely_fast:
                np.multiply(value, dt, self.buf) # buf = value * dt
                np.add(self.buf, self.accum, self.accum) # accum += buf
            else:
                self.buf = value * dt
                self.accum += self.buf

            self.needs_normalization = True
            self.accum_mass += dt

        if self.max_window and self.accum_mass > self.max_window:
            self.accum = self.max_window * self.get_value()
            self.accum_mass = self.max_window

        MAX_MASS = 100 # TODO: watch np.max(self.accum) instead
        # Do not let pass too much before normalization
        if self.accum_mass > MAX_MASS:
            self.get_value()

    def get_value(self):
        if self.accum is None:
            raise ValueError('No value given yet.')
        if self.needs_normalization:
            # In the case dt=0 for the first sample
            if self.accum_mass > 0:
                ratio = 1.0 / self.accum_mass
            else:
                ratio = 1.0
            if self.extremely_fast:
                np.multiply(ratio, self.accum, self.result)
            else:
                self.result = ratio * self.accum
            self.needs_normalization = False
        return self.result

    def __call__(self):
        return self.get_value()

    def get_mass(self):
        return self.accum_mass

Expectation = ExpectationFast
#Expectation = ExpectationSlow
