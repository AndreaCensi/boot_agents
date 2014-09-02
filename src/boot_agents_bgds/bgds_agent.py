from bgds import BGDSEstimator, BGDSPredictor, smooth2d
from blocks import Sink, check_timed_named
from boot_agents.deriv.deriv_agent import DerivAgent
from bootstrapping_olympics import (PredictingAgent, ServoingAgent, 
    UnsupportedSpec)
from contracts import check_isinstance, contract
import numpy as np

__all__ = ['BGDSAgent']
 

class BGDSAgent(DerivAgent, PredictingAgent, ServoingAgent):
    '''
        scales: list of floats, represents the scales at which the 
                sensels are analyzed. 0=raw data, 1= convolved with sigma=1.
    '''
    #@contract(scales='list[>=1](number,>=0)')
    def __init__(self):
        pass
    
    def init(self, boot_spec):
        # TODO: do the 1D version
        shape = boot_spec.get_observations().shape()
        if len(shape) > 2:
            msg = 'BGDSagent only work with 2D or 1D signals.'
            raise UnsupportedSpec(msg)

        min_width = np.min(shape)
        if min_width <= 5:
            msg = ('BGDSagent thinks this shape is too'
                   'small to compute gradients: %s' % str(shape))
            raise UnsupportedSpec(msg)

        self.is2D = len(boot_spec.get_observations().shape()) == 2
        self.is1D = len(boot_spec.get_observations().shape()) == 1

        self.count = 0
        self.bgds_estimator = BGDSEstimator()

#         self.model = None
#         self.y_disag = Expectation()
#         self.y_disag_s = Expectation()
#         self.u_stats = []
#         
        self.last_y = None

    
    @contract(returns=Sink)
    def get_learner_u_y_y_dot(self):
        """ 
            Returns a Sink that receives dictionaries
            dict(y=..., y_dot=..., u=...)
        """
        
        class MySink(Sink):
            def __init__(self, agent):
                self.agent = agent
            def reset(self):
                pass
            def put(self, value, block=False, timeout=None):  # @UnusedVariable
                check_timed_named(value)
                (_, (name, x)) = value  # @UnusedVariable
                check_isinstance(x, dict)
                y = x['y_dot']
                y_dot = x['y']
                u = x['u']
                
                u=u.astype('float32')
                y=y.astype('float32')
                y_dot=y_dot.astype('float32')
                
                self.agent.update(u=u, y=y,y_dot=y_dot, w=1.0)

        return MySink(self) 
    
    def update(self, u, y, y_dot, w):
        self.bgds_estimator.update(u=u.astype('float32'),
                                   y=y.astype('float32'),
                                   y_dot=y_dot.astype('float32'),
                                   dt=w)
        self.last_y = y
        
    
    def merge(self, agent2):
        assert isinstance(agent2, BGDSAgent)
        self.bgds_estimator.merge(agent2.bgds_estimator)

        # TODO: implement this separately
    
#         if False: # and self.is2D and self.count > MINIMUM_FOR_PREDICTION:
#             # TODO: do for 1D
#             if self.count % 200 == 0 or self.model is None:
#                 self.info('Updating BGDS model.')
#                 self.model = self.bgds_estimator.get_model()
# 
#             gy = self.bgds_estimator.last_gy
#             y_dot_est = self.model.estimate_y_dot(y, u, gy=gy)
#             y_dot_corr = y_dot_est * y_dot_sync
#             self.y_disag.update(np.maximum(-y_dot_corr, 0))
#             self.y_disag_s.update(np.sign(y_dot_corr))
# 
#             u_est = self.model.estimate_u(y, y_dot_sync, gy=gy)
# 
#             data = {'u': u,
#                     'u_est': u_est,
#                     'timestamp': obs.time,
#                     'id_episode': obs.id_episode
#             }
#             self.u_stats.append(data)

#          u_est = self.model.estimate_u(y, y_dot_sync, gy=self.bgds_estimator)
#          self.u_stats.append()
#            
    def publish(self, pub):
        with pub.subsection('model')as sub:
            self.bgds_estimator.publish(sub)

        if False and self.is2D:  # TODO: implement separately
            sec = pub.section('preprocessing')
            sec.array_as_image('last_y', self.last_y, filter='scale')
            example = np.zeros(self.last_y.shape)
            example.flat[150] = 1
            example_smooth = create_scales(example, self.scales)
            sec.array_as_image('example_smooth', example_smooth)

            if True: #if self.count > MINIMUM_FOR_PREDICTION:
                sec = pub.section('reliability')
                sec.array_as_image('y_disag',
                                   self.y_disag.get_value(), filter='posneg')
                sec.array_as_image('y_disag_s',
                                   self.y_disag_s.get_value(), filter='posneg')

        if False:  # XXX
            self.publish_u_stats(pub.section('u_stats'))

    def publish_u_stats(self, pub):
        from reprep import MIME_PDF
        from reprep.plot_utils import x_axis_set, y_axis_set

        T = len(self.u_stats)
        K = 2  # FIXME: change this
        u_act = np.zeros((T, K))
        u_est = np.zeros((T, K))
        u_mis = np.zeros((T, K))
        u_suc = np.zeros((T, K))
        time = np.zeros(T)
        num_episode = np.zeros(T, 'int')
        id_episode2num = {}
        num2id_episode = {}
        id_episode2start = {}
        # cmd2faults = {}
        for t, stats in enumerate(self.u_stats):
            u_act[t, :] = stats['u']
            u_est[t, :] = stats['u_est']
            time[t] = stats['timestamp']
            id_ep = stats['id_episode']
            if not id_ep in id_episode2num:
                id_episode2num[id_ep] = len(id_episode2num)
                id_episode2start[id_ep] = time[t]
                num2id_episode[id_episode2num[id_ep]] = id_ep
            num_episode[t] = id_episode2num[id_ep]

        s = ""
        for k, v in id_episode2num.items():
            s += '%s: %s\n' % (k, v)
        pub.text('episodes', s)
        with pub.plot('num_episode') as pylab:
            pylab.plot(num_episode, '-')
            pylab.xlabel('index')
            pylab.ylabel('num\_episode')

        for id_episode, num in id_episode2num.items():
            print id_episode
            S = pub.section('Episode:%s' % id_episode)
            # times for this episode
            et = num_episode == num
            # normalize from 0
            e_timestamps = time[et]
            log_start = e_timestamps[0]
            e_timestamps -= log_start
            cmd2color = {0: 'g', 1: 'b'}

            episode_bounds = (18, 60)
            markersize = 2
            with S.plot('mis', figsize=(8, 2), mime=MIME_PDF) as pylab:
                for k in range(K):
#                    scale = 7
#                    u_mis_smooth = scipy.signal.convolve(u_mis[et, k], 
#                     np.ones(scale) / scale,
#                                                         mode='same')
                    pylab.plot(e_timestamps, u_mis[et, k],  # u_mis_smooth,
                               '%s-' % cmd2color[k], label='u[%d]' % k,
                               markersize=markersize)
                x_axis_set(pylab, episode_bounds[0], episode_bounds[1])

            with S.plot('success', figsize=(8, 2), mime=MIME_PDF) as pylab:
                pylab.plot(e_timestamps, e_timestamps * 0, 'k--')
                pylab.plot(e_timestamps, np.ones(len(e_timestamps)), 'k--')
                for k in range(K):
                    pylab.plot(e_timestamps, u_suc[et, k],
                               '%s-' % cmd2color[k], label='cmd #%d' % k)
                y_axis_set(pylab, -0.05, 1.05)
                x_axis_set(pylab, episode_bounds[0], episode_bounds[1])
                pylab.legend(loc='lower right')

            for k in range(K):
                with S.plot('commands_%d' % k, figsize=(8, 2),
                            mime=MIME_PDF) as pylab:
                    pylab.plot(e_timestamps, u_act[et, k], 'y.',
                               label='actual', markersize=3)
                    plot_with_colors(pylab, e_timestamps,
                                     u_est[et, k], u_act[et, k],
                                     markersize=markersize)
                    y_axis_set(pylab, -2, 2)
                    x_axis_set(pylab, episode_bounds[0], episode_bounds[1])

    def get_predictor(self):
        if self.count == 0:
            msg = 'No observation processed yet.'
            raise ValueError(msg)
        model = self.bgds_estimator.get_model()
        return BGDSPredictor(model)


    def get_servo_system(self):
        # TODO: implement
        raise NotImplementedError(type(self))


def plot_with_colors(pylab,
                     timestamps, values, values_giving_colors,
                     value2color=[(-1, 'b'), (0, 'k'), (+1, 'r')], **kwargs):
    for (u, col) in value2color:
        which = values_giving_colors == u
        pylab.plot(timestamps[which], values[which], '%s.' % col,
                                    label='estimated', **kwargs)


@contract(y='array[HxW]', scales='list[M](0|(float,>=0))',
                  returns='array[Hx(W*M)]')
def create_scales(y, scales):
    data = []
    for s in scales:
        if s == 0:
            data.append(y)
        else:
            data.append(smooth2d(y, s))
    return np.hstack(data)
