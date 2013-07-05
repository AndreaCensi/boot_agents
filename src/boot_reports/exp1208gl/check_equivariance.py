# from .. import logger
# from boot_agents.bdse.model.bdse_tensors import get_M_from_P_T_Q_alt
# from boot_reports.latex.bds import get_bds_M, get_bds_T, get_bds_P, get_bds_Q
# from boot_reports.latex.load import load_report_phase
# from bootstrapping_olympics import get_boot_config
# from bootstrapping_olympics.library.robots import EquivRobot
# from collections import namedtuple
# from contracts import contract
# from numpy.testing import assert_allclose
# from reprep import Report, posneg
# import numpy as np
# 
# 
# Case = namedtuple('Case', 'id_robot M T A P')
# cases = [] 
# 
# def read_data(id_set, id_robot, id_agent, config):
# 
#     robot = config.robots.instance(id_robot)
#             
#     r = load_report_phase(id_set, id_agent, id_robot, 'learn')
#     M = get_bds_M(r)
#     T = get_bds_T(r)
#     n = T.shape[0]
#     P = get_bds_P(r)
#     Q = get_bds_Q(r)
#     
#     if isinstance(robot, EquivRobot):
#         nuisance = robot.obs_nuisances[0]
#         try:
#             A = nuisance.A
#         except:
#             perm = nuisance.perm 
#             A = np.zeros((n, n))
#             for i in range(n):
#                 A[i, perm[i]] = 1
#     else:
#         A = np.eye(n) 
#         
#     logger.info('Recomputing M')
#     M = get_M_from_P_T_Q_alt(P, T, Q)
#         
#     case = Case(id_robot=id_robot, M=M, T=T, A=A, P=P)
#     return case 
# 
# 
# def check_linear_tran(id_set='bv1bds4'):
#     id_agent = 'bdse1'
#     original = 'Se0Vrb1ro'
#     transformed = ['Yrl1Se0Vrb1ro', 'Yrl2Se0Vrb1ro', 'Yrp1Se0Vrb1ro']
#     
#     transformed = ['Yrs1Se0Vrb1ro', 'Yrp1Se0Vrb1ro', 'Yrl1Se0Vrb1ro'] 
# 
#     config_dir = '/Users/andrea/scm/boot11_env/src/bvapps/bdse1/config/' 
#     config = get_boot_config()
#     vconfig = VehiclesConfig
#     config.load(config_dir)
#     vconfig.load(config_dir)
#     
#     rep = Report('myrep')
#     allr = [original] + transformed
#     
#     cases = map(lambda x: read_data(id_set, x, id_agent, config), allr)
#     
#     c0 = cases[0]  # first is the reference one
#     for c in cases[1:]:  # skip first
#         print(c.id_robot)
#         sec = rep.section(c.id_robot)
#         f = sec.figure('A')
#         f.data_rgb('A', posneg(c.A))
#         f = sec.figure('tensors', cols=3)
#         K = c.M.shape[2]
#         for k in range(K):
#             f.data_rgb('M%d' % k, posneg(c.M[:, :, k]))
#         
#         for k in range(K):
#             f.data_rgb('T%d' % k, posneg(c.T[:, :, k]))
#         
#         f.data_rgb('P', posneg(c.P))
#         P2 = conj(c0.P, c.A)
#         
#         f = sec.figure('Pcomp')
#         f.data_rgb('P2', posneg(P2), caption='A P0 A* (should be equal to P)')
#         
#         err = c.P - P2
#         f.data_rgb('err', posneg(err), caption='P - A P0 A* (should be 0)')
#         
#         perc = 0.04
#         with f.plot('err_plot') as pylab:
#             plot_against(pylab, c.P, P2, perc=perc)
#             pylab.xlabel('P')
#             pylab.ylabel('A P0 A*') 
#             
#         f = sec.figure('Tcomp')
#         for k in range(K):
#             T0 = c0.T[:, :, k]
#             T1 = c.T[:, :, k]
#             AT0A = conj(T0, c.A)
#         
#             with f.plot('%s' % k) as pylab:
#                 plot_against(pylab, T1, AT0A, perc=perc)
#                 pylab.xlabel('T1_k')
#                 pylab.ylabel('A T0_k A*')
#                 
#         cases = [ 
#                     ('conjugation1', conjugation1),
#                     # ('conjugation2', conjugation2),
#                     ('conjugation3', conjugation3),
#                     # ('conjugation4', conjugation4)
#                 ]
#         
#         for name, func in cases:
#             f = sec.figure('Mcomp-%s' % name)
#             for k in range(K):
#                 M0 = c0.M[:, :, k]
#                 M1 = c.M[:, :, k]
#                 AM0A = func(M0, c.A)
#             
#                 with f.plot('%s' % k) as pylab:
#                     plot_against(pylab, M1, AM0A, perc=perc)
#                     pylab.xlabel('M1_k')
#                     pylab.ylabel('A M0_k A*')
#         
#     rep.to_html('check_linear_tran.html')
# 
# def plot_against(pylab, x, y, perc=0.1):
#     x = np.array(x.flat)
#     y = np.array(y.flat)
#     sel = np.random.rand(x.size) < perc
#     x = x[sel]
#     y = y[sel]
#     pylab.plot(x, y, 's', markersize=0.1)
#     pylab.axis('equal')
#     
#     
# @contract(M1='array[NxN]', X='array[NxN]', M2='array[NxN]',
#           returns='array[NxN]')
# def prod(M1, X, M2):
#     return np.dot(M1, np.dot(X, M2))
# 
# 
# #    MP = np.tensordot(M, P, axes=(1, 0))
# #    print('M', M.shape)
# #    print('P', P.shape)
# #    print('MP', MP.shape)
# #    print('MP2', MP2.shape)
# #    printm('M', M, 'P', P, 'Q', Q, 'MP', MP)
# #    MPQ = np.tensordot(MP, Q, axes=(2, 0))
# #    print('MPQ', MPQ.shape)
# #    return MPQ
#     
# 
# def prod_test():
#     n, k = 10, 3
#     X = np.random.rand(n, n)
#     M1 = np.eye(n)
#     M2 = np.eye(n)
#     assert_allclose(X, prod(M1, X, M2))
# 
# def conj(X, A):
#     """ A X A* """
#     M1 = A
#     M2 = A.T
#     return prod(M1, X, M2)
# 
# def conjugation1(X, A):
#     """ A X A^-1 """
#     M1 = A
#     M2 = np.linalg.inv(A)
#     return prod(M1, X, M2)
# #
# # def conjugation2(X, A):
# #    """ A^-1 X A """
# #    M1 = np.linalg.inv(A)
# #    M2 = A
# #    return prod(M1, X, M2)
# 
# def conjugation3(X, A):
#     """ A^-T X AT """
#     M1 = np.linalg.inv(A).T
#     M2 = A.T
#     return prod(M1, X, M2)
# 
# # def conjugation4(X, A):
# #    """ A.T  X A^-T """
# #    M1 = A.T
# #    M2 = np.linalg.inv(A).T
# #    return prod(M1, X, M2)
