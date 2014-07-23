from contracts import contract

from geometry import mds, euclidean_distances
import numpy as np
from streamels import UnsupportedSpec, ValueFormats

from bootstrapping_olympics.interfaces.agent import LearningAgent, BasicAgent
from blocks.library.timed.checks import check_timed_named
from blocks.interface import Sink


__all__ = ['SymbolsStats']


class SymbolsStats(BasicAgent, LearningAgent):

    @contract(window='int,>=1')
    def __init__(self, window):
        self.window = window

    def init(self, boot_spec):
        streamels = boot_spec.get_observations().get_streamels()
        if ((streamels.size != 1) or
            (streamels[0]['kind'] != ValueFormats.Discrete)):
            msg = 'I expect the observations to be symbols (one discrete var).'
            raise UnsupportedSpec(msg)

        self.lower = streamels[0]['lower']
        nsymbols = int(1 + (streamels[0]['upper'] - self.lower))
        if nsymbols > 10000:
            msg = ('Are you sure you want to deal with %d symbols?'
                   % nsymbols)
            raise Exception(msg)

        from boot_agents.utils import SymbolsStatistics

        self.y_stats = SymbolsStatistics(nsymbols, window=self.window)


    def get_learner_as_sink(self):
        class LearnSink(Sink):
            def __init__(self, y_stats, lower):
                self.y_stats = y_stats
                self.lower = lower
            def reset(self):
                pass
            def put(self, value, block=True, timeout=None):  # @UnusedVariable
                check_timed_named(value)
                timestamp, (signal, obs) = value  # @UnusedVariable
                if not signal in ['observations', 'commands']:
                    msg = 'Invalid signal %r to learner.' % signal
                    raise ValueError(msg)
                
                if signal == 'observations':
                    symbol = obs[0].item() - self.lower
                    self.y_stats.update(symbol, dt=1.0)
                
        return LearnSink(self.y_stats, self.lower)
    
#     def process_observations(self, obs):
#         y = obs['observations']
#         dt = obs['dt'].item()
#         symbol = y[0].item() - self.lower
#         self.y_stats.update(symbol, dt)

    def publish(self, pub):
        self.y_stats.publish(pub.section('y_stats'))

        def display_solution(name, caption, R, S):
            D = euclidean_distances(S)
            sec = pub.section(name)
            sec.text('info', caption)
            with sec.plot('S') as pl:
                pl.plot(S[0, :], S[1, :], 'k.')

            if S.shape[0] == 3:
                import mpl_toolkits.mplot3d.axes3d as p3

                for angle in [0, 45, 90]:
                    with sec.plot('S3d%s' % angle) as pl:
                        fig = pl.gcf()
                        ax = p3.Axes3D(fig)
                        ax.view_init(30, angle)
                        col = S[2, :]
                        ax.scatter(S[0, :], S[1, :], S[2, :], c=col)
                        ax.set_xlabel('X')
                        ax.set_ylabel('Y')
                        ax.set_zlabel('Z')
                        fig.add_axes(ax)

            if False:  # TODO: use cheaper solution for point cloud as in
                    # calib
                with sec.plot('D_vs_R') as pl:
                    pl.plot(D.flat, R.flat, 'k.')
                    pl.xlabel('D')
                    pl.ylabel('R')

        deltas = range(1, self.window)
        for d in deltas:
            R = self.y_stats.get_transition_matrix(delta=d)
            S = self.get_embedding(R, ndim=2)
            display_solution('delta%d' % d, 'Using delta=%d' % d, R, S)

            S3 = self.get_embedding(R, ndim=3)
            display_solution('delta%dS3' % d, 'Using delta=%d' % d, R, S3)

        Ts = [self.y_stats.get_transition_matrix(delta=d) for d in deltas]
        T = np.array(Ts)
        R = np.sum(T, axis=0)
        S = self.get_embedding(R, ndim=2)
        display_solution('deltasum', 'Summing all of them', R, S)
        S3 = self.get_embedding(R, ndim=3)
        display_solution('deltasum3', 'Summing all of them', R, S3)

    @contract(R='array[NxN]', ndim='int,>=2')
    def get_embedding(self, R, ndim):
        D = 1 - R
        D = D * np.pi / D.max()
        np.fill_diagonal(D, 0)

#        if pub is not None:
#            P = D * D
#            B = double_center(P)
#            plot_spectrum(pub, 'B', B)

        S = mds(D, ndim=ndim)
        return S

