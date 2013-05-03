from boot_agents.diffeo.diffeo_display import angle_legend
from boot_agents.diffeo.diffeo_visualization import scalaruncertainty2rgb
from boot_agents.diffeo.diffeomorphism2d import Diffeomorphism2D
from reprep import Report
import numpy as np
import pickle
import warnings
 
 
class Diffeomorphism2DContinuous(Diffeomorphism2D):
    def __init__(self, d, variance=None):
        self.d = d
        
        if variance is None:
            self.variance = np.ones((d.shape[0], d.shape[1]))
        else:
            assert variance.shape == d.shape[:2]
            assert np.isfinite(variance).all()
            
            # Normalize
            self.variance_max = np.max(variance)
            self.variance = 1 - variance.astype('float32') / self.variance_max
            
        # Make them immutable
        self.variance = self.variance.copy()
        self.variance.setflags(write=False)
        self.d = self.d.copy()
        self.d.setflags(write=False)

    def display(self, report, full=False, nbins=100):
        """ Displays this diffeomorphism. """
        # Inherit method from superclass
        Diffeomorphism2D.display(self, report, full, nbins)

        warnings.warn('removed part of visualization')
        if False:
            self.display_mesh(report=report, nbins=nbins)
            
    def display_mesh(self, report, nbins):
        # Additional plots for the continuous diffeo
        Y, X = np.meshgrid(range(self.d.shape[1]), range(self.d.shape[0]))
        
        xq = (self.d[:, :, 0] - X)
        xqf = xq.reshape(self.d.size / 2)
        yq = (self.d[:, :, 1] - Y)
        yqf = yq.reshape(self.d.size / 2)

        f0 = report.figure(cols=6)
        with f0.plot('angle_legend', caption='Angle legend') as pylab:
            pylab.imshow(angle_legend((20, 20), 0))

        with f0.plot('quiver', caption='quiver plot') as pylab:
            pylab.quiver(Y, X, yq, xq)
            
        if hasattr(self, 'plot_ranges') and hasattr(self, 'variance_max'):
            if 'uncertainty_max' in self.plot_ranges:
                
                if hasattr(self, 'variance_max'):
                    varmax_text = '(variance max %s)' % self.variance_max
                else:
                    varmax_text = ''
                    
                variance_max = self.plot_ranges['uncertainty_max']
                
                uncert = self.get_scalar_info()
                variance = (1 - uncert) * self.variance_max
                uncert_new = 1 - variance / variance_max 
                info_rgb = scalaruncertainty2rgb(uncert_new)
                f0.data_rgb('uncert_fixed', info_rgb,
                            caption='Uncertainty (green=sure, red=unknown %s)' % varmax_text)
                
            
        with f0.plot('x_hist', caption='x displacement') as pylab:
            pylab.hist(xqf, nbins)
            
        with f0.plot('y_hist', caption='y displacement') as pylab:
            pylab.hist(yqf, nbins)
            
        

def make_report(learners):
    print('make_report(learners) in diffeomorphism2d_continuous is used')
    for i, name in enumerate(learners):
        # init report
        report = Report(learners[i])
        
        learner = pickle.load(open(name))
        diffeo = learner.estimators[0].summarize()
        learner.estimators[0].show_areas(report, diffeo.d)
        cmd = learner.command_list[0]
#        pdb.set_trace()
        report.text('learner' + str(i), name)
        report.text('cmd' + str(i), str(cmd))
        diffeo.display(report, nbins=500)
        
        # Save report
        report.to_html(learners[i] + '.html')
#    from boot_agents.diffeo import diffeomorphism2d_continuous as d2c
