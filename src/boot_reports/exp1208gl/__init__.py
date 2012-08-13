from boot_reports.latex.load import load_report_phase
from reprep import Report
from bootstrapping_olympics.configuration.master import BootOlympicsConfig
from vehicles.configuration.master import VehiclesConfig
from bootstrapping_olympics.library.robots.equiv_robot import EquivRobot
from collections import namedtuple
from reprep.graphics.filter_posneg import posneg
import numpy as np

Case = namedtuple('Case', 'id_robot M T A P')
cases = [] 

def read_data(id_set, id_robot, id_agent, config):
    from boot_reports.latex import *

    robot = config.robots.instance(id_robot)
            
    r = load_report_phase(id_set, id_agent, id_robot, 'learn')
    M = get_bds_M(r)
    n = M.shape[0]
    T = get_bds_T(r)
    P = get_bds_P(r)
    
    if isinstance(robot, EquivRobot):
        nuisance = robot.obs_nuisances[0]
        try:
            A = nuisance.A
        except:
            perm = nuisance.perm 
            A = np.zeros((n, n))
            for i in range(n):
                A[i, perm[i]] = 1
    else:
        A = np.eye(n) 
    case = Case(id_robot=id_robot, M=M, T=T, A=A, P=P)
    return case 


def check_linear_tran(id_set):

    id_agent = 'bdse1'
    original = 'Se0Vrb1ro'
    transformed = ['Yrl1Se0Vrb1ro', 'Yrl2Se0Vrb1ro', 'Yrp1Se0Vrb1ro']

    config_dir = '/Users/andrea/scm/boot11_env/src/bvapps/bdse1/config/' 
    config = BootOlympicsConfig
    vconfig = VehiclesConfig
    config.load(config_dir)
    vconfig.load(config_dir)
    
    rep = Report('myrep')
    allr = [original] + transformed
    
    cases = map(lambda x: read_data(id_set, x, id_agent, config), allr)
    
    c0 = cases[0]
    for c in cases:
        print(c.id_robot)
        sec = rep.section(c.id_robot)
        f = sec.figure('A')
        f.data_rgb('A', posneg(c.A))
        f = sec.figure('tensors', cols=3)
        K = c.M.shape[2]
        for k in range(K):
            f.data_rgb('M%d' % k, posneg(c.M[:, :, k]))
        
        for k in range(K):
            f.data_rgb('T%d' % k, posneg(c.T[:, :, k]))
        
        f.data_rgb('P', posneg(c.P))
        P2 = conj(c0.P, c.A)
        
        f = sec.figure('Pcomp')
        f.data_rgb('P2', posneg(P2), caption='A P0 A* (should be equal to P)')
        
        err = c.P - P2
        f.data_rgb('err', posneg(err), caption='P - A P0 A* (should be 0)')
        
        with f.plot('err_plot') as pylab:
            plot_against(pylab, c.P, P2, perc=0.1)
            pylab.xlabel('P')
            pylab.ylabel('A P0 A*') 
            
        f = sec.figure('Tcomp')
        for k in range(K):
            T0 = c0.T[:, :, k]
            T1 = c.T[:, :, k]
            AT0A = conj(T0, c.A)
        
            with f.plot('%s' % k) as pylab:
                plot_against(pylab, T1, AT0A, perc=0.1)
                pylab.xlabel('T1_k')
                pylab.ylabel('A T0_k A*')
                
        f = sec.figure('Mcomp')
        for k in range(K):
            M0 = c0.M[:, :, k]
            M1 = c.M[:, :, k]
            AM0A = conjugation2(M0, c.A)
        
            with f.plot('%s' % k) as pylab:
                plot_against(pylab, M1, AM0A, perc=1)
                pylab.xlabel('M1_k')
                pylab.ylabel('A M0_k A*')
        
    rep.to_html('check_linear_tran.html')

def plot_against(pylab, x, y, perc=0.1):
    x = np.array(x.flat)
    y = np.array(y.flat)
    sel = np.random.rand(x.size) < perc
    x = x[sel]
    y = y[sel]
    pylab.plot(x, y, 's', markersize=0.1)
    pylab.axis('equal')
    

def conj(X, A):
    """ A X A* """
    return np.dot(A, np.dot(X, A.T))

def conjugation(X, A):
    """ A X A^-1 """
    return np.dot(A, np.dot(X, np.linalg.inv(A)))
def conjugation2(X, A):
    """ A X A^-1 """
    return np.dot(np.linalg.inv(A), np.dot(X, A))
