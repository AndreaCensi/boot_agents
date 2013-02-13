from boot_agents.diffeo.diffeomorphism2d import Diffeomorphism2D
import numpy as np
from boot_agents.diffeo.diffeo_display import diffeo_stats, angle_legend
from reprep import Report
import pickle

class Diffeomorphism2DContinuous(Diffeomorphism2D):
    def __init__(self, d, variance=None):
        
        self.d = d
        
        if variance is None:
            self.variance = np.ones((d.shape[0], d.shape[1]))
        else:
            assert variance.shape == d.shape[:2]
            assert np.isfinite(variance).all()
            self.variance = variance.astype('float32')
            
        # Make them immutable
        self.variance = self.variance.copy()
        self.variance.setflags(write=False)
        self.d = self.d.copy()
        self.d.setflags(write=False)

    def display(self, report, full=False, nbins=100):
        """ Displays this diffeomorphism. """
        stats = diffeo_stats(self.d)
        angle = stats.angle
        norm = stats.norm
        
        norm_rgb = self.get_rgb_norm()
        angle_rgb = self.get_rgb_angle()
        info_rgb = self.get_rgb_info()
        
        Y, X = np.meshgrid(range(self.d.shape[1]), range(self.d.shape[0]))
        
        xq = (self.d[:, :, 0] - X)
        xqf = xq.reshape(self.d.size / 2)
        yq = (self.d[:, :, 1] - Y)
        yqf = yq.reshape(self.d.size / 2)
#        pdb.set_trace()
        
        f0 = report.figure(cols=9)
        with f0.plot('angle_legend', caption='Angle legend') as pylab:
            pylab.imshow(angle_legend((20, 20), 0))
        
        f = report.figure(cols=9)
        f.data_rgb('norm_rgb', norm_rgb,
                    caption="Norm(D). white=0, blue=maximum (%.2f). " % np.max(norm))
        f.data_rgb('phase_rgb', angle_rgb,
                    caption="Phase(D).")
        
        f.data_rgb('var_rgb', info_rgb,
                    caption='Uncertainty (green=sure, red=unknown)')

        with f.plot('norm_hist', caption='histogram of norm values') as pylab:
            pylab.hist(norm.flat, nbins)

        angles = np.array(angle.flat)
        valid_angles = angles[np.logical_not(np.isnan(angles))]
        with f.plot('angle_hist', caption='histogram of angle values '
                    '(excluding where norm=0)') as pylab:
            pylab.hist(valid_angles, nbins)

        with f.plot('var_hist', caption='histogram of certainty values') as pylab:
            pylab.hist(self.variance.flat, nbins)

        with f.plot('x_hist', caption='x displacement') as pylab:
            pylab.hist(xqf, nbins)
            
        with f.plot('y_hist', caption='y displacement') as pylab:
            pylab.hist(yqf, nbins)
            
            
        with f.plot('quiver', caption='quiver plot') as pylab:
            pylab.quiver(Y, X, yq, xq)

def make_report(learners):
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
