from . import np, contract
from ..utils import Expectation, outer, MeanCovariance
from boot_agents.utils import PredictionStats
from contracts import new_contract
from geometry import printm
from numpy.linalg.linalg import LinAlgError
import scipy.linalg


@contract(M='array[KxNxN]', y='array[N]', u='array[K]')
def bds_dynamics(M, y, u):
    y_dot = np.dot(u, np.dot(M, y))
    return y_dot


# TODO: put in contracts
@new_contract
@contract(x='array')
def array_finite(x):
    return np.isfinite(x).all()

__all__ = ['BDSEstimator2', 'bds_dynamics']


class BDSEstimator2:

    def __init__(self, min_count_for_prediction=100):
        self.T = Expectation()
        self.uu = Expectation()
        self.yy = Expectation()
        self.yy_inv = None
        self.yy_inv_needs_update = True
        self.M = None
        self.MP = None
        self.M_needs_update = True

        self.y_dot_noise = MeanCovariance()

        self.y_dot_stats = MeanCovariance()
        self.y_dot_pred_stats = MeanCovariance()

        self.y_dot_pred_stats = PredictionStats('y_dot', 'y_dot_pred')
        self.Py_dot_pred_stats = PredictionStats('Py_dot', 'Py_dot_pred')
#
#        self.y_dots_stats = MeanCovariance()
#        self.Py_dots_stats = MeanCovariance()

        self.fits1 = Expectation()
        self.fits2 = Expectation()

        self.u2y2 = Expectation()

        self.count = 0

        self.min_count_for_prediction = min_count_for_prediction

    @contract(u='array[K],K>0,array_finite',
              y='array[N],N>0,array_finite',
              y_dot='array[N],array_finite', dt='>0')
    def update(self, u, y, y_dot, dt):
        if np.linalg.norm(u) == 0:
            self.y_dot_noise.update(y_dot, dt)
            return

        self.num_commands = u.size
        self.num_sensels = y.size
        Tk = outer(u, outer(y, y_dot))

        self.T.update(Tk, dt)
        self.yy.update(outer(y, y), dt)
        self.uu.update(outer(u, u), dt)

        uy = outer(u, y)
        self.u2y2.update(uy * uy, dt)

        self.yy_inv_needs_update = True
        self.M_needs_update = True
        self.count += 1

        if self.count > self.min_count_for_prediction:
            self.update_prediction_stats(y, y_dot, u, dt)

    def update_prediction_stats(self, y, y_dot, u, dt):
        T = self.get_T()
        uu_inv = np.linalg.pinv(self.get_uu()) # XXX: change this
        un = np.dot(uu_inv, u)

        A = np.tensordot(un, T, ([0], [0]))
        Py_dot_pred = np.dot(A.T, y)
        Py_dot = np.dot(self.get_yy(), y_dot)
        error1 = np.abs(np.sign(Py_dot) - np.sign(Py_dot_pred))
        error2 = np.abs(Py_dot - Py_dot_pred)
        self.fits1.update(error1)
        self.fits2.update(error2)

        use_old_version = self.count % 50 != 0
        M = self.get_M(rcond=1e-5, use_old_version=use_old_version)
        y_dot_pred = bds_dynamics(M, y, u)

#        def printstats(name, x):
#            print('stats for %15s: [%10f, %10f]' %
#                      (name, np.min(x), np.max(x)))
#
#        printstats('Py_dot', Py_dot)
#        printstats('Py_dot_pred', Py_dot_pred)
#        printstats('y_dot', y_dot)
#        printstats('y_dot_pred', y_dot_pred)

        # Assuming y in [-1,+1], we have the following bound
        delta_y_bound = 1 - (-1)
        # If we do this in 1 time step, then the derivative is
        y_dot_bound = np.abs(delta_y_bound) / dt
        # Therefore, we bound the prediction
        y_dot_pred = np.clip(y_dot_pred, -y_dot_bound, +y_dot_bound)

#        printstats('y_dot_pred(clip)', y_dot_pred)

        self.y_dot_pred_stats.update(y_dot, y_dot_pred, dt)
        self.Py_dot_pred_stats.update(Py_dot, Py_dot_pred, dt)

        self.last_y_dot = y_dot
        self.last_y_dot_pred = y_dot_pred
        self.last_Py_dot = Py_dot
        self.last_Py_dot_pred = Py_dot_pred

    @contract(returns='array_finite')
    def get_yy_inv(self, rcond=1e-5):
        if not hasattr(self, 'yy_inv_rcond'):
            setattr(self, 'yy_inv_rcond', 0)
        if self.yy_inv_needs_update or (rcond != self.yy_inv_rcond):
            self.yy_inv_needs_update = False
            yy = self.yy.get_value()
            self.yy_inv = np.linalg.pinv(yy, rcond)
            self.yy_inv_rcond = rcond
        return self.yy_inv

    @contract(returns='array_finite')
    def get_M(self, rcond=1e-5, use_old_version=False):
        if not hasattr(self, 'M_rcond'):
            setattr(self, 'M_rcond', 0)
        needs_update = self.M_needs_update or (self.M_rcond != rcond)
        if self.M is None or (needs_update and not use_old_version):
            self.M_needs_update = False
            T = self.get_T()
            if self.MP is None:
                self.MP = np.zeros(T.shape, T.dtype)
            for k in range(self.num_commands):
                yy = self.get_yy()
                Tk = T[k, :, :]
                try:
                    Mk = scipy.linalg.solve(yy, Tk) #@UndefinedVariable
                except LinAlgError:
                    # yy is singular  
                    # print('Using pseudoinverse, rcond=%s' % rcond)
                    #yy_pinv = self.get_yy_inv(rcond)
                    yy_pinv = np.linalg.inv(np.eye(yy.shape[0]) * rcond + yy)
                    Mk = np.dot(yy_pinv, Tk)
                self.MP[k, :, :] = Mk.T # note transpose

            uu_inv = np.linalg.pinv(self.get_uu()).astype(self.MP.dtype)
            self.M = np.tensordot(uu_inv, self.MP, ([0], [0]))
        return self.M

    def get_M2(self, rcond=1e-5):
        T = self.get_T()
        M2 = np.empty_like(T)
        M2info = np.empty_like(M2)
#        yy = self.get_yy()
        u2y2 = self.u2y2.get_value()
        for k in range(self.num_commands):
            for v in range(self.num_sensels):
                M2[k, :, v] = T[k, v, :] / u2y2[k, v]
                M2info[k, :, v] = u2y2[k, v]
#            Tk = T[k, :, :]
#            M2[k, :, :] = Tk / uy[k, :, :]  # note transpose
        return M2, M2info

    def get_T(self):
        return self.T.get_value()

    def get_yy(self):
        return self.yy.get_value()

    def get_uu(self):
        return self.uu.get_value()

    def publish(self, pub):
        if self.T.get_mass() == 0:
            pub.text('warning',
                     'No samples obtained yet -- not publishing anything.')
            return
        #params = dict(filter=pub.FILTER_POSNEG, filter_params={'skim':2})
        params = dict(filter=pub.FILTER_POSNEG, filter_params={})

        rcond = 1e-2
        T = self.get_T()
        M = self.get_M(rcond)
        yy_inv = self.get_yy_inv(rcond)
        yy = self.get_yy()

        if False:
            # TODO: computation of usefulness
            y_dots_corr = self.y_dots_stats.get_correlation()
            n = T.shape[2]
            measured_corr = y_dots_corr[:n, n:].diagonal() # upper right
            try:
                var_noise = self.y_dot_noise.get_covariance().diagonal()
                var_prediction = y_dots_corr.diagonal()[n:]
                invalid = var_noise == 0
                var_noise[invalid] = 0
                var_prediction[invalid] = 1
                corrected_corr = measured_corr * np.sqrt(1 + var_noise /
                                                         var_prediction)
                with pub.plot('correlation') as pylab:
                    pylab.plot(measured_corr, 'bx', label='raw')
                    pylab.plot(corrected_corr, 'rx', label='corrected')
                    pylab.axis((-1, n, -0.1, 1.1))
                    pylab.ylabel('correlation')
                    pylab.xlabel('sensel')
                    pylab.legend()
            except:
                pass # XXX: 

        def pub_tensor(name, V):
            section = pub.section(name)
            for i in range(V.shape[0]):
                section.array_as_image('%s%d' % (name, i), V[i, :, :],
                                       **params)

        pub_tensor('T', T)
        pub_tensor('M', M)

        if T.shape[0] == 2:
            # Only for 2 commands so far
            Tortho, Q = orthogonalize(T) #@UnusedVariable
            Tortho_norm = normalize(Tortho, yy)
            pub_tensor('Tortho', Tortho)
            pub_tensor('Tortho_norm', Tortho_norm)

        if False:
            M2, M2info = self.get_M2()
            pub_tensor('M2', M2)
            pub_tensor('M2info', M2info)

        try:
            self.y_dot_noise.publish(pub.section('y_dot_noise'))
        except:
            pass

        self.y_dot_pred_stats.publish(pub.section('y_dot_pred_stats'))
        self.Py_dot_pred_stats.publish(pub.section('Py_dot_pred_stats'))

        pub.array_as_image('yy', self.get_yy(), **params)
        pub.array_as_image('yy_inv', yy_inv, **params)
        pub.array_as_image('uu', self.get_uu(), **params)

        with pub.plot('yy_svd') as pylab:
            u, s, v = np.linalg.svd(yy) #@UnusedVariable
            s /= s[0]
            pylab.semilogy(s, 'bx-')
            pylab.semilogy(np.ones(s.shape) * rcond, 'k--')

        sec = pub.section('prediction')
        with sec.plot('fits1') as pylab:
            q = self.fits1.get_value()
            pylab.plot(q, 'x')

        with sec.plot('fits2') as pylab:
            q = self.fits2.get_value()
            pylab.plot(q, 'x')

        with sec.plot('last_values_Py_dot') as pylab:
            pylab.plot(self.last_Py_dot, 'kx-', label='actual')
            pylab.plot(self.last_Py_dot_pred, 'go-', label='pred')

        with sec.plot('last_values_y_dot') as pylab:
            pylab.plot(self.last_y_dot, 'kx-', label='actual')
            pylab.plot(self.last_y_dot_pred, 'go-', label='pred')

        with sec.plot('last_values_Py_dot_v') as pylab:
            x = self.last_Py_dot
            y = self.last_Py_dot_pred
            pylab.plot(x, y, '.')
            pylab.xlabel('observation')
            pylab.ylabel('prediction')

        with sec.plot('last_values_y_dot_v') as pylab:
            x = self.last_y_dot
            y = self.last_y_dot_pred
            pylab.plot(x, y, '.')
            pylab.xlabel('observation')
            pylab.ylabel('prediction')

    def publish_compact(self, pub, rcond=1e-2):
        if self.T.get_mass() == 0:
            pub.text('warning',
                     'No samples obtained yet -- not publishing anything.')
            return

        params = dict(filter=pub.FILTER_POSNEG, filter_params={})

        T = self.get_T()
        M = self.get_M(rcond=rcond)
        yy = self.get_yy()

        def pub_tensor(name, V):
            section = pub.section(name)
            for i in range(V.shape[0]):
                section.array_as_image('%s%d' % (name, i), V[i, :, :],
                                       **params)

        pub_tensor('T', T)
        pub_tensor('M', M)

        pub.array_as_image('yy', self.get_yy(), **params)

        with pub.plot('yy_svd') as pylab:
            u, s, v = np.linalg.svd(yy) #@UnusedVariable
            s /= s[0]
            pylab.semilogy(s, 'bx-')
            pylab.semilogy(np.ones(s.shape) * rcond, 'k--')


def normalize(T, P):
    # XXX: again
    #R = cov2corr(P)
    # Tn, Q = normalize(T, yy)
    Tn = np.empty_like(T)
    for i in range(Tn.shape[0]):
        Tn[i, :, :] = T[i, :, :] * P * P
    return Tn


@contract(T='array[KxNxN]', returns='tuple(array[KxNxN], array[KxK])')
def orthogonalize(T):
    if T.shape[0] != 2:
        raise ValueError('Sorry, not implemented yet')
    K = T.shape[0]
    N = np.zeros((K, K))
    Z = np.empty_like(T)
    for k in range(K):
        N[k, k] = 1.0 / np.linalg.norm(T[k, :, :])
        Z[k, :, :] = T[k, :, :] * N[k, k]

    proj = (Z[0, :, :] * Z[1, :, :]).sum()
    print('Projection: %s' % proj)
    W = np.empty_like(T)
    W[0, :, :] = Z[0, :, :] - proj * Z[1, :, :]
    W[1, :, :] = proj * Z[0, :, :] + Z[1, :, :]
    for k in range(K):
        W[k, :, :] = W[k, :, :] / np.linalg.norm(W[k, :, :])

    Q = np.zeros((K, K))
    Q[0, :] = [1, -proj]
    Q[1, :] = [proj, 1]
    A = np.dot(Q, N)
    printm('Q', Q, 'N', N, 'A', A)
    return W, A


