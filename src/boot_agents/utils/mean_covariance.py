from contracts import contract

from astatsa.mean_covariance import MeanCovariance as MeanCovarianceBase
from boot_agents.misc_utils import y_axis_positive, y_axis_extra_space
import numpy as np
from reprep.plot_utils import style_ieee_fullcol_xy


__all__ = ['MeanCovariance']


class MeanCovariance(MeanCovarianceBase):
    
    @contract(publish_information='bool')
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

        pub.text('data', 'Num samples: %s' % self.mean_accum.get_mass())

        if Ey.size > 1:
            with pub.plot('expectation') as pylab:
                style_ieee_fullcol_xy(pylab)
                pylab.plot(Ey, 's', label='expectation')
                pylab.plot(y_max, 's', label='max')
                pylab.plot(y_min, 's', label='min')
                y_axis_extra_space(pylab)
                pylab.legend()
            
            stddev = np.sqrt(P.diagonal())
            with pub.plot('stddev') as pylab:
                style_ieee_fullcol_xy(pylab)
                pylab.plot(stddev, 's', label='stddev')
                y_axis_extra_space(pylab)
                pylab.legend()
                
            self.print_odd_ones(pub, P, perc=50, ratio=0.2)

            from boot_agents.misc_utils.tensors_display import pub_tensor2_cov
            pub_tensor2_cov(pub, 'covariance', P)
            # pub.array_as_image('covariance', P)
            

            # TODO: get rid of this?
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
        else:
            stats = ""
            stats += 'min: %g\n' % y_min
            stats += 'mean: %g\n' % Ey
            stats += 'max: %g\n' % y_min
            stats += 'std: %g\n' % np.sqrt(P)
            pub.text('stats', stats)

    def print_odd_ones(self, pub, P, perc=50, ratio=0.2):
        odd, okay = get_odd_measurements(P, perc, ratio)
        
        f = pub.figure()
        pub.text('okay', '%s' % list(okay))
        pub.text('odd', '%s' % list(odd))
        
        std = np.sqrt(P.diagonal())
        with f.plot('odd') as pylab:
            style_ieee_fullcol_xy(pylab)
            pylab.plot(odd, std[odd], 'rs')
            pylab.plot(okay, std[okay], 'gs')
            y_axis_extra_space(pylab)
            

def get_odd_measurements(P, perc, ratio):
    std = np.sqrt(P.diagonal())
    odd, okay = find_odd(std, perc, ratio)
    return odd, okay
            
def find_odd(values, perc, ratio):
    """ Writes the ones that have smaller variance.
        Defined as ratio * value at percentile """
    v = np.percentile(values, perc)
    threshold = v * ratio
    odd, = np.nonzero(values < threshold)
    okay, = np.nonzero(values >= threshold) 
    assert len(odd) + len(okay) == len(values)
    return odd, okay
    
        
        
        
        
