# from reprep.plot_utils import x_axis_set, y_axis_set
# from boot_agents.misc_utils import style_1d_sensel_func, BV1Style
# import numpy as np
# import astatsa
# 
# class PredictionStats(astatsa.PredictionStats):
# 
#     def publish(self, pub):
#         if self.num_samples == 0:
#             pub.text('warning',
#                      'Cannot publish anything as I was never updated.')
#             return
# 
#         pub.text('stats', 'Num samples: %s' % self.num_samples)
# 
#         R = self.get_correlation()
#         
#         pub.array('R', R)
#         pub.array('last_a', self.last_a)
#         pub.array('last_b', self.last_b)
# 
#         if R.ndim == 1:
#             with pub.plot('correlation') as pylab:
#                 pylab.plot(R, 'k.')
#                 pylab.axis((0, R.size, -1.1, +1.1))
# 
#             with pub.plot('last') as pylab:
#                 pylab.plot(self.last_a, 'g.', label=self.label_a)
#                 pylab.plot(self.last_b, 'm.', label=self.label_b)
#                 a = pylab.axis()
#                 m = 0.1 * (a[3] - a[2])
#                 pylab.axis((a[0], a[1], a[2] - m, a[3] + m))
#                 pylab.legend()
#             
#             with pub.plot('vs') as pylab:
#                 pylab.plot(self.last_a, self.last_b, '.')
#                 pylab.xlabel(self.label_a)
#                 pylab.ylabel(self.label_b)
#                 pylab.axis('equal')
#             
#             # TODO: remove    
#             with pub.plot('comp_balanced') as pylab:
#                 pylab.plot(self.last_a, 'g.', label=self.label_a)  # XXX: colors
#                 pylab.plot(self.last_b, 'm.', label=self.label_b)
#                 comparison_balanced(pylab, self.last_a, self.last_b, perc=0.9, M=1.1)
#                 style_1d_sensel_func(pylab, n=R.size, y_max=1)
# 
#             with pub.plot('comp_balanced_lines') as pylab:
#                 pylab.plot(self.last_a, 'g-', label=self.label_a)  # XXX: colors
#                 pylab.plot(self.last_b, 'm-', label=self.label_b)
#                 comparison_balanced(pylab, self.last_a, self.last_b, perc=0.9, M=1.1)
#                 style_1d_sensel_func(pylab, n=R.size, y_max=1)
#                 
#             with pub.plot('correlationB', figsize=BV1Style.figsize) as pylab:
#                 pylab.plot(R, **BV1Style.dots_format)
#                 style_1d_sensel_func(pylab, n=R.size, y_max=1)
#                 
#         elif R.ndim == 2:
#             pass
#             pub.text('warning', 'not implemented yet for ndim == 2')
# 
#         self.Ea.publish(pub.section('%s_stats' % self.label_a))
#         self.Eb.publish(pub.section('%s_stats' % self.label_b))
# 
# 
# 
# def comparison_balanced(pylab, a, b, perc=0.95, M=1.1):
#     """
#         Computes 1-perc and perc percentiles, and fits only as much.
#         
#     """
#     values = np.hstack((a, b))
#     
#     level = np.percentile(np.abs(values), perc * 100)
#     limit = M * level
#     y_axis_set(pylab, -limit, limit)
#     
#     n = a.size
#     x_axis_set(pylab, -1, n)
# 
# 
#     
