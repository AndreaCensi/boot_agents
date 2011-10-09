from . import BGDSEstimator, smooth2d, np, contract
from ..simple_stats import ExpSwitcher
from ..utils import DerivativeBox, Expectation, RemoveDoubles
from collections import namedtuple
from reprep import MIME_PDF
import scipy.signal
from bootstrapping_olympics import UnsupportedSpec


__all__ = ['BGDSAgent']

MINIMUM_FOR_PREDICTION = 200 # XXX

class BGDSAgent(ExpSwitcher):
    '''
        Skip: only consider every $skip observations.
        
        scales: list of floats, represents the scales at which the 
                sensels are analyzed. 0=raw data, 1= convolved with sigma=1.
    '''
    @contract(scales='list[>=1](number,>=0)')
    def __init__(self, beta, skip=1, scales=[0], fixed_dt=0):
        ExpSwitcher.__init__(self, beta)
        self.skip = skip
        self.scales = scales
        self.fixed_dt = fixed_dt
               
    def init(self, boot_spec):
        # TODO: do the 1D version
        if len(boot_spec.get_observations().shape()) != 2:
            raise UnsupportedSpec('Can only work with 2D signals.')

        ExpSwitcher.init(self, boot_spec)
        self.count = 0
        self.y_deriv = DerivativeBox()
        self.bgds_estimator = BGDSEstimator()  
        
        self.model = None
        self.y_disag = Expectation()
        self.y_disag_s = Expectation()
        self.u_stats = []
        self.last_y0 = None
        self.last_y = None

        self.rd = RemoveDoubles(0.5)

    def process_observations(self, obs):
        dt = float(obs['dt']) 
        u = obs['commands']
        y0 = obs['observations']
        episode_start = obs['episode_start']

        self.count += 1 
        if self.count % self.skip != 0:
            return
        
        if self.fixed_dt: 
            # dt is not reliable sometime
            # you don't want to give high weight to higher dt samples.
            dt = 1 
            
        
        self.rd.update(y0)
        if not self.rd.ready():
            return
        
        y = create_scales(y0, self.scales)
            
        if episode_start:
            self.y_deriv.reset()
            return        

        self.y_deriv.update(y, dt)
        
        if not self.y_deriv.ready():
            return
        
        y_sync, y_dot_sync = self.y_deriv.get_value()

        self.bgds_estimator.update(u=u.astype('float32'),
                                   y=y_sync.astype('float32'),
                                   y_dot=y_dot_sync.astype('float32'),
                                   dt=dt)
        self.last_y0 = y0
        self.last_y = y
        

        if self.count > MINIMUM_FOR_PREDICTION:
            if self.count % 200 == 0 or self.model is None:
                self.info('Updating BGDS model.')
                self.model = self.bgds_estimator.get_model()
            
            gy = self.bgds_estimator.last_gy
            y_dot_est = self.model.estimate_y_dot(y, u, gy=gy)
            y_dot_corr = y_dot_est * y_dot_sync
            self.y_disag.update(np.maximum(-y_dot_corr, 0))
            self.y_disag_s.update(np.sign(y_dot_corr))
            
            u_est = self.model.estimate_u(y, y_dot_sync, gy=gy)
            
            data = {'u': u,
                    'u_est': u_est,
                    'timestamp': obs.time,
                    'id_episode': obs.id_episode
            }
            self.u_stats.append(data) 
            
#            u_est = self.model.estimate_u(y, y_dot_sync, gy=self.bgds_estimator)
#            self.u_stats.append()
#            
    def publish(self, publisher):
        if self.count < 10: 
            self.info('Skipping publishing as count=%d' % self.count)
            return
        
        self.bgds_estimator.publish(publisher)

        sec = publisher.section('preprocessing')
        sec.array_as_image('last_y0', self.last_y0, filter='scale')
        sec.array_as_image('last_y', self.last_y, filter='scale')
        example = np.zeros(self.last_y.shape)
        example.flat[150] = 1 
        example_smooth = create_scales(example, self.scales) 
        sec.array_as_image('example_smooth', example_smooth)
        
        if self.count > MINIMUM_FOR_PREDICTION:
            sec = publisher.section('reliability')
            sec.array_as_image('y_disag',
                               self.y_disag.get_value(), filter='posneg')
            sec.array_as_image('y_disag_s',
                               self.y_disag_s.get_value(), filter='posneg')
            
        if True:
            self.publish_u_stats(publisher.section('u_stats'))
            
    def publish_u_stats(self, pub):
        T = len(self.u_stats)
        print('Obtained %d obs' % T)
        K = 2 # FIXME: change this
        u_act = np.zeros((T, K))
        u_est = np.zeros((T, K))
        u_mis = np.zeros((T, K))
        u_suc = np.zeros((T, K))
        time = np.zeros(T)
        num_episode = np.zeros(T, 'int')
        id_episode2num = {} 
        num2id_episode = {}
        id_episode2start = {}
        cmd2faults = {}
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
#          
        for k in range(K):
            sec = pub.section('u%d' % k) 
            
            good_data = np.array(range(T)) > 15000
            for u in np.unique(u_act[:, k]):
                which, = np.nonzero((u_act[:, k] == u) & good_data)
                with sec.plot('u=%s' % u, figsize=(6, 3), mime=MIME_PDF) as pylab:
#                    hist_max = 400
                    pylab.hist(u_est[which, k], 100)
                    a = pylab.axis()
#                    pylab.plot([u, u], [0, a[3]], 'r--')
                    pylab.axis((-2, 2, a[2], a[3]))

            mis0 = -u_est[:, k] * u_act[:, k]
#            u_mis[:, k] = np.maximum(-u_est[:, k] * u_act[:, k], 0)
#            u_mis[:, k] = np.abs(u_est[:, k] - u_act[:, k])

            u_mis[:, k] = (np.maximum(-u_est[:, k] * u_act[:, k], 0) + 
                                     np.abs(u_est[:, k]) * (u_act[:, k] == 0))
                           
            u_suc[:, k] = count_positive(-mis0, interval=400)
            faults = detect_faults(num_episode, time, u_mis[:, k],
                                    scale=7, threshold=0.75,
                                    minimum_spacing=5)
            cmd2faults[k] = faults
            self.info('For u[%d], %d faults' % (k, len(faults)))
            s = ""
            for fault in faults:
                id_ep = num2id_episode[fault.num_episode]
                s += ('Episode %s, timestamp %s (%s)\n' % 
                       (id_ep, fault.timestamp,
                        fault.timestamp - id_episode2start[id_ep]))
            sec.text('faults', s)

            graphics = dict(markersize=1)
            
            with sec.plot('time') as pylab:
                plot_with_colors(pylab, np.array(range(T)),
                 u_est[:, k], u_act[:, k], **graphics)

#                pylab.plot(u_act[:, k], 'k.', **graphics)
#                pylab.plot(u_est[:, k], 'b.', **graphics)
                
            with sec.plot('mis') as pylab:
                pylab.plot(u_mis[:, k], 'r.', **graphics)
                for fault in faults:
                    pylab.plot(fault.index, fault.detect, 'kx')
                    
            with sec.plot('success') as pylab:
                pylab.plot(u_suc[:, k], 'r.', **graphics)
                set_y_axis(pylab, -0.05, 1.05)
            
            with sec.plot('mis0') as pylab:
                pylab.plot(mis0, 'r.', **graphics)
                
        for id_episode, num in id_episode2num.items():
            print id_episode
#            if id_episode != '20100615_234934': continue
            S = pub.section('Episode:%s' % id_episode)
            # times for this episode
            et = num_episode == num
            # normalize from 0
            e_timestamps = time[et]
            log_start = e_timestamps[0] 
            e_timestamps -= log_start
            cmd2color = {0:'g', 1:'b'}
            
            episode_bounds = (18, 60)
            markersize = 2
            with S.plot('mis', figsize=(8, 2), mime=MIME_PDF) as pylab:
                for k in range(K):
#                    scale = 7
#                    u_mis_smooth = scipy.signal.convolve(u_mis[et, k], np.ones(scale) / scale,
#                                                         mode='same')
                    pylab.plot(e_timestamps, u_mis[et, k], #u_mis_smooth,
                               '%s-' % cmd2color[k], label='u[%d]' % k,
                               markersize=markersize)
                #plot_fault_lines(pylab, cmd2faults, num, log_start, cmd2color)
#                pylab.legend()
                set_x_axis(pylab, episode_bounds[0], episode_bounds[1])

                
            with S.plot('success', figsize=(8, 2), mime=MIME_PDF) as pylab:
                pylab.plot(e_timestamps, e_timestamps * 0, 'k--')
                pylab.plot(e_timestamps, np.ones(len(e_timestamps)), 'k--')
                for k in range(K):
                    pylab.plot(e_timestamps, u_suc[et, k],
                               '%s-' % cmd2color[k], label='cmd #%d' % k)
                set_y_axis(pylab, -0.05, 1.05) 
                set_x_axis(pylab, episode_bounds[0], episode_bounds[1])
                pylab.legend(loc='lower right')
     
            for k in range(K):
                with S.plot('commands_%d' % k, figsize=(8, 2), mime=MIME_PDF) as pylab:
                    pylab.plot(e_timestamps, u_act[et, k], 'y.', label='actual', markersize=3)
                    plot_with_colors(pylab, e_timestamps,
                                     u_est[et, k], u_act[et, k], markersize=markersize)
                    set_y_axis(pylab, -2, 2)
                    #plot_fault_lines(pylab, {k:cmd2faults[k]}, num, log_start, cmd2color)
                    set_x_axis(pylab, episode_bounds[0], episode_bounds[1])


def plot_with_colors(pylab,
                     timestamps, values, values_giving_colors,
                     value2color=[(-1, 'b'), (0, 'k'), (+1, 'r')], **kwargs):
    for (u, col) in value2color:
        which = values_giving_colors == u
        pylab.plot(timestamps[which], values[which], '%s.' % col,
                                    label='estimated', **kwargs)
                 
     
def set_y_axis(pylab, y_min, y_max):
    a = pylab.axis()
    pylab.axis([a[0], a[1], y_min, y_max])

def set_x_axis(pylab, x_min, x_max):
    a = pylab.axis()
    pylab.axis([x_min, x_max, a[2], a[3]])
    
def plot_fault_lines(pylab, cmd2faults, num_episode, episode_time_start, cmd2color):
    a = pylab.axis()
    for k, faults in cmd2faults.items():
        xs = []
        ys = []    
        for fault in faults:
            if fault.num_episode != num_episode: continue
            t = fault.timestamp - episode_time_start
            xs.extend([t, t, None])
            ys.extend([a[2], a[3], None])
        pylab.plot(xs, ys, '%s-' % cmd2color[k])
               

# TODO: remove all of this
Fault = namedtuple('Fault', 'index,num_episode,timestamp,detect,detect_smooth')
def detect_faults(episode, timestamp, signal, scale, threshold, minimum_spacing):
    x = scipy.signal.convolve(signal, np.ones(scale) / scale, mode='same')
    assert len(x) == len(signal)
    maxima = local_minima(-x) & (x > threshold)
    locations, = np.nonzero(maxima)
    print('Total %d faults considering' % len(locations))
    faults = []
    order = np.argsort(-x[locations])
    for index in order:
        l = locations[index]
        cur_time = timestamp[l]
        for l2, _, other_time, _, _ in faults:
            spacing = np.abs(other_time - cur_time) 
            if spacing < minimum_spacing:
                #print('Skipping %d because %d is close  (%f)' % 
                #      (l, l2, spacing))
                break  
        else:
            faults.append(Fault(l,
                       episode[l],
                       timestamp[l], signal[l], x[l]))
    print('Total %d faults found.' % len(faults))
    faults = sorted(faults, key=lambda x: x[0])
    return faults

def local_minima(x):
    return ((x <= np.roll(x, +1, 0)) & 
            (x <= np.roll(x, -1, 0)))

def count_positive(x, interval):
    n = x.size
    y = np.zeros(n)
    for i in range(n):
        a = max(0, i - interval)
        b = min(n, i + interval)
        part = x[a:b]
        y[i] = (part >= 0).sum() * 1.0 / (b - a)
    return y

def create_scales(y='array[HxW]', scales='list[M](float,>=0)',
                  returns='array[Hx(W*M)]'):
    data = []
    for s in scales:
        if s == 0:
            data.append(y)
        else:
            data.append(smooth2d(y, s))
    return np.hstack(data)
