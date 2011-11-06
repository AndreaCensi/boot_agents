from . import np
from ..simple_stats import ExpSwitcher
from ..utils import DerivativeBox, MeanCovariance, scale_score
# FIXME: dependency to remove
from geometry import inner_product_embedding
from bootstrapping_olympics import UnsupportedSpec

__all__ = ['Embed']

class Embed(ExpSwitcher):
     
    def init(self, boot_spec):
        ExpSwitcher.init(self, boot_spec)
        if len(boot_spec.get_observations().shape()) != 1:
            raise UnsupportedSpec('I assume 1D signals.')
            
        self.y_stats = MeanCovariance()
        self.z_stats = MeanCovariance()
        self.count = 0
        self.y_deriv = DerivativeBox()
        
    def process_observations(self, obs):
        y = obs['observations']
        dt = obs['dt'].item()
        self.y_stats.update(y, dt)

        self.y_deriv.update(y, dt)
        if self.y_deriv.ready():
            y, y_dot = self.y_deriv.get_value()
            self.z_stats.update(np.abs(y_dot), dt)
            self.count += 1
    
    def publish(self, pub):
        if self.count < 10:
            pub.text('warning', 'Too early to publish anything.') 
            return
        
        R = self.z_stats.get_correlation()
        Dis = discretize(-R, 2)
        np.fill_diagonal(Dis, 0)
        pub.array_as_image('Dis', Dis)
        R = R * R
        C = np.maximum(R, 0)

        S = inner_product_embedding(Dis, 2)
        for i in range(R.shape[0]):
            R[i, i] = np.NaN
            C[i, i] = np.NaN
        pub.array_as_image('R', R)
        pub.array_as_image('C', C)
        with pub.plot(name='S') as pylab:
            pylab.plot(S[0, :], S[1, :], '.')
        self.z_stats.publish(pub.section('z_stats'))
    

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

