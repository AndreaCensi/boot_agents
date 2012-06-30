from . import np, contract
#from ..utils import Expectation, outer, MeanCovariance
from numpy.linalg.linalg import LinAlgError
import scipy.linalg
from boot_agents.utils.expectation import Expectation
from boot_agents.utils.mean_covariance import MeanCovariance
from boot_agents.utils.outer import outer
from boot_agents.bdse.model.bdse_model import BDSEmodel
from geometry.formatting import printm


class BDSEEstimator:
    """
         M^s_vi   (N) x (N x K)
         N^s_i    (N) x (K)
         T^svi    (NxNxK)
         U^si     (NxK)
    
    """

    def __init__(self, min_count_for_prediction=100):
        self.T = Expectation()
        self.U = Expectation()
        self.y_stats = MeanCovariance()
        self.u_stats = MeanCovariance()

    @contract(u='array[K],K>0,array_finite',
              y='array[N],N>0,array_finite',
              y_dot='array[N],array_finite', w='>0')
    def update(self, y, u, y_dot, w=1.0):
        self.n = y.size
        self.k = u.size # XXX: check

        self.y_stats.update(y, w)
        self.u_stats.update(u, w)
        # remove mean
        u_n = u - self.u_stats.get_mean()
        y_n = y - self.y_stats.get_mean()
        # form tensor
        T_k = outer(outer(y_n, y_dot), u_n) # XXX could be other way
        assert T_k.shape == (self.n, self.n, self.k)
        U_k = outer(y_dot, u_n)
        assert U_k.shape == (self.n, self.k)
        # update tensor
        self.T.update(T_k, w)
        self.U.update(U_k, w)

    def get_model(self):
        T = self.T.get_value()
        U = self.U.get_value()
        P = self.y_stats.get_covariance()
        Q = self.u_stats.get_covariance()

        P_inv = np.linalg.pinv(P)
        Q_inv = np.linalg.pinv(Q)

        printm('Tshape', T.shape)
        printm('T', T)
        printm('U', U)
        printm('P', P)
        printm('Q', Q)
        printm('Q_inv', Q_inv)
        printm('P_inv', P_inv)

        #print 'T', T.shape, 'P', P_inv.shape
        TP_inv = np.tensordot(T, P_inv, axes=(0, 0))
        TP_inv = np.transpose(TP_inv, (0, 2, 1))

        #print 'TP_inv', TP_inv.shape, Q_inv.shape
        M = np.tensordot(TP_inv, Q_inv, axes=(2, 0))
        #print U.shape, Q_inv.shape
        UQ_inv = np.tensordot(U, Q_inv, axes=(1, 0))

        Myav = np.tensordot(M, self.y_stats.get_mean(), axes=(1, 0))

        N = UQ_inv - Myav
        return BDSEmodel(M, N)

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

