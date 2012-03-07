from . import np
from ..simple_stats import ExpSwitcher
from ..utils import DerivativeBox, MeanCovariance, scale_score
from bootstrapping_olympics import UnsupportedSpec
from geometry import double_center, inner_product_embedding, mds
from reprep.plot_utils import style_ieee_halfcol_xy
# FIXME: dependency to remove

__all__ = ['Embed']


class Embed(ExpSwitcher):

    def __init__(self, statistic='y_corr', scale_score=False, **kwargs):
        ExpSwitcher.__init__(self, **kwargs)
        self.statistic = statistic
        self.scale_score = False

    def init(self, boot_spec):
        ExpSwitcher.init(self, boot_spec)
        if len(boot_spec.get_observations().shape()) != 1:
            raise UnsupportedSpec('I assume 1D signals.')

        self.y_stats = MeanCovariance()
        self.y_dot_stats = MeanCovariance()
        self.y_dot_sgn_stats = MeanCovariance()
        self.y_dot_abs_stats = MeanCovariance()

        self.count = 0
        self.y_deriv = DerivativeBox()

    def get_similarity(self, which):
        if which == 'y_corr':
            return self.y_stats.get_correlation()
        if which == 'y_dot_corr':
            return self.y_dot_stats.get_correlation()
        if which == 'y_dot_sgn_corr':
            return self.y_dot_sgn_stats.get_correlation()
        if which == 'y_dot_abs_corr':
            return self.y_dot_abs_stats.get_correlation()

        raise ValueError()
        #check_contained(statistic, self.statistics, 'statistic')

    def process_observations(self, obs):
        y = obs['observations']
        dt = obs['dt'].item()

        self.y_deriv.update(y, dt)
        if self.y_deriv.ready():
            y, y_dot = self.y_deriv.get_value()
            self.y_stats.update(y, dt)
            self.y_dot_stats.update(y_dot, dt)
            self.y_dot_sgn_stats.update(np.sign(y_dot), dt)
            self.y_dot_abs_stats.update(np.abs(y_dot), dt)
            self.count += 1

    def get_S(self, dimensions=2, pub=None):
        similarity = self.get_similarity(self.statistic)
        if pub is not None:
            pub.array_as_image('similarity', similarity,
                               caption='Similarity statistic')
            plot_spectrum(pub, 'similarity', similarity)

        if self.scale_score:
            R = scale_score(similarity).astype('float32')
            R = R / R.max()
            if pub is not None:
                pub.array_as_image('scale_score', R)

        else:
            R = similarity

        D = 1 - R
        D = D * np.pi / D.max()
        np.fill_diagonal(D, 0)

        if pub is not None:
            #            pub.array_as_image('D', D)
            P = D * D
            B = double_center(P)
            #            plot_spectrum(pub, 'D', D)
            #            plot_spectrum(pub, 'P', P)
            plot_spectrum(pub, 'B', B)

        S = mds(D, ndim=2)
#        S = inner_product_embedding(similarity, 3)
#        S = S[1:3, :]
        return S

    def get_S_discrete(self, dimensions=2, pub=None):
        R = self.y_dot_abs_stats.get_correlation()
        Dis = discretize(-R, 2)
        np.fill_diagonal(Dis, 0)
        R = R * R
        C = np.maximum(R, 0)

        if pub is not None:
            pub.array_as_image('Dis', Dis)
            pub.array_as_image('R', R)
            pub.array_as_image('C', C)

        S = inner_product_embedding(Dis, 2)
#        for i in range(R.shape[0]):
#            R[i, i] = np.NaN
#            C[i, i] = np.NaN
        return S

    def publish(self, pub):
        if self.count < 10:
            pub.text('warning', 'Too early to publish anything.')
            return

        pub.text('info', 'Using statistics: %s' % self.statistic)

        if False: # TODO: make option
            S = self.get_S_discrete(2, pub=pub.section('computation'))
        else:
            S = self.get_S(2, pub=pub.section('computation'))

        with pub.plot(name='S') as pylab:
            style_ieee_halfcol_xy(pylab)
            pylab.plot(S[0, :], S[1, :], 's')

        with pub.plot(name='S_joined') as pylab:
            style_ieee_halfcol_xy(pylab)
            pylab.plot(S[0, :], S[1, :], '-')

        self.y_stats.publish(pub.section('y_stats'))
        self.y_dot_stats.publish(pub.section('y_dot_stats'))
        self.y_dot_sgn_stats.publish(pub.section('y_dot_sgn_stats'))
        self.y_dot_abs_stats.publish(pub.section('y_dot_abs_stats'))


def discretize(M, w):
    X = np.zeros(M.shape, dtype='float32')
    for i in range(M.shape[0]):
        score = scale_score(M[i, :])
        which, = np.nonzero(score <= w)
        X[i, which] += 1
    for j in range(M.shape[0]):
        score = scale_score(M[:, j])
        which, = np.nonzero(score <= w)
        X[which, j] += 1
    return X


def plot_spectrum(pub, name, matrix):
    from scipy.linalg import eigh
    S, _ = eigh(matrix) # returns the eigenvalues reversed
    eigs = S[::-1]
    with pub.plot('%s-eigs' % name, caption='Spectrum of %s' % name) as pylab:
        style_ieee_halfcol_xy(pylab)
        svd_plot(pylab, eigs)


def svd_plot(pylab, svds, rcond=None):
    svds = svds / svds[0]
    pylab.semilogy(svds, 'sk')
    if rcond is not None:
        pylab.semilogy(np.ones(svds.shape) * rcond, 'k--')


def inner_product_embedding2(C, ndim):
    n = C.shape[0]
    eigvals = (n - ndim, n - 1)
    from scipy.linalg import eigh

    S, V = eigh(C, eigvals=eigvals)
    coords = V.T
    for i in range(ndim):
        coords[i, :] = coords[i, :] * np.sqrt(S[i])

    S, V = eigh(C) # returns the eigenvalues reversed
    S = S[::-1]
    return coords, S



#        similarity = np.maximum(0, similarity)
#
#        def compress(x):
#            x = np.clip(x, -1, +1)
#            return np.sin(x * (np.pi / 2))
#
#        def decompress(y):
#            y = np.clip(y, -1, +1)
#            return  np.arcsin(y) / (np.pi / 2)

#        similarity = decompress(similarity)

        #@contract(C='array[NxN]', ndim='int,>0,K', returns='array[KxN]')

#
#        R = np.empty_like()
#        for i in range(R.shape[0]):
#            R[i, :] = scale_score(similarity())
#            
