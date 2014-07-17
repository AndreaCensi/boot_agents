from contracts import contract, new_contract
import numpy as np
import itertools
import warnings
from .interface import BDSEServoInterface
from bootstrapping_olympics import ServoAgentInterface

__all__ = ['ServoLongTermMotion', 'BDSEServoLongTerm', 'myexp']


class ServoLongTermMotion():
    
    @contract(R='array[NxN](>=0,<=1)')
    def __init__(self, R):
        self.R = R.astype('float32')
        self.Rw = np.sum(self.R, axis=1)
        
    def get_visibility(self):
        return np.mean(self.Rw)
    
    def get_R(self):
        return self.R
    
    @contract(returns='tuple(array[N], array[N](>=0,<=1))')
    def predict_inverse(self, y1):
        y1 = y1.astype('float32')
        y0 = np.dot(self.R, y1)
        return y0, self.Rw
    
    def predict(self, y0):
        raise NotImplementedError()
    
    def display(self, report):
        f = report.figure(cols=8)
        report.data('R', self.R).display('scale').add_to(f, caption='R')
        with f.plot('Rw') as pylab:
            pylab.plot(self.Rw, '.')
            # report.data('Rw', self.Rw).display('scale').add_to(f, caption='Rw')
        report.text('visibility', str(self.get_visibility()))
        
new_contract('ServoLongTermMotion', ServoLongTermMotion)
        
        
class BDSEServoLongTerm(BDSEServoInterface):
    """ This uses long-term prediction. """
    
    # @contract(plans='dict(tuple(tuple, float):ServoLongTermMotion)')
    def __init__(self, grid, gain=0.045, min_visibility=0.2):
        self.plans = None
        self.bdse_model = None
        self.grid = grid
        self.gain = gain
        self.min_visibility = min_visibility
    
    def init(self, boot_spec):
        self.boot_spec = boot_spec
        self.commands_spec = boot_spec.get_commands()    
        
    def set_model(self, model):
        M = model.M
        K = M.shape[2]
        cmds = BDSEServoLongTerm.get_cmds(K=K, **self.grid)
        print('creating %d plans' % len(cmds))
        plans = {}
        for plan in cmds:
            cmd, t = plan
            # wrong wrong wrong
            # no, actually ok
            u = np.array(cmd) * t
            A = np.einsum("abc,c -> ab", M, u)
            R = myexp(A)
            motion = ServoLongTermMotion(R)
            
            vis = motion.get_visibility()
            if vis < self.min_visibility:
                print('%s, %s -> vis %s < %s' % (cmd, t, vis, self.min_visibility))
            else:
                plans[plan] = motion
            
        self.plans = plans
        
    @staticmethod
    def get_cmds(K, mode, grid_max, grid_n):
        assert mode == 'uniform'
        cmds = []
        
        ncmds_max = 500
        ncmds_from_n = lambda n: n ** K
        while ncmds_from_n(grid_n) > ncmds_max:
            print('Grid %s gives too many cmds %s' % 
                  (grid_n, ncmds_from_n(grid_n)))
            grid_n -= 1
        
        assert grid_n >= 3
        
        base = np.linspace(-grid_max, grid_max, grid_n)

        xs = list(itertools.product(base, repeat=K))
        
        print('Using %d grid -> %d commands' % (grid_n, len(xs)))
        assert len(xs) == ncmds_from_n(grid_n)
        
        for x in xs:
            x = np.array(x)
            t = np.linalg.norm(x)
            x = x / t
            cmds.append((tuple(x), t))
        return cmds
    
    def _get_ordered_plans(self):
        def order(x):
            (plan, t), _ = x  # @UnusedVariable
            return np.linalg.norm(np.array(plan)) * t
        xs = list(self.plans.items())
        xs.sort(key=order)
        return xs
    
    def display(self, report):  # @UnusedVariable
        for i, (plan, action) in enumerate(self._get_ordered_plans()):
            # cmd, time = plan
            with report.subsection('plan%d' % i) as sub:
                sub.text('plan', str(plan))
                action.display(sub) 
        
    @contract(y0='array[N]', y_goal='array[N]', limit='int,>=1')
    def find_best(self, y0, y_goal, limit):
        compare = False
        if compare:
            D = get_distance_map(y0, y_goal)
        sols = []
        for plan in self.plans:
            w2 = self.get_cost(y0, y_goal, plan)
            if compare:
                w1 = self.get_cost_distance_matrix(D, plan)
                print 'e %5.10f %10f %10f' % (w1 - w2, w1, w2)
            sols.append((w2, plan))
        sols.sort(key=lambda x: x[0])
        
        sols2 = sols[:limit]
        # print sols2
        return [x[1] for x in sols2]
    
    def get_cost(self, y0, y_goal, plan):
        """ Returns the cost for a plan. """
        assert plan in self.plans
        motion = self.plans[plan]
        # Predict y_goal
        y_goal_back, known = motion.predict_inverse(y_goal)        
        # only look at the ones that are known
        diff = y0 - y_goal_back
        error = np.sum(np.abs(diff) * known)
        errorw = error / np.sum(known)
        return errorw

    def get_cost_distance_matrix(self, D, plan):
        """ 
            Returns the cost for a plan, using the distance matrix 
        
        """
        assert plan in self.plans
        motion = self.plans[plan]
        R = motion.get_R()
        DR = D * R
        w = np.sum(DR) / np.sum(R)
        return w
        
    @contract(goal='array')
    def set_goal_observations(self, goal):
        self.y_goal = goal.copy()

    def process_observations(self, obs):
        self.y = obs['observations'].copy()

    def choose_commands(self):
        sols = self.find_best(self.y, self.y_goal, limit=1)
        plan = sols[0]
        cmd, _ = plan
        cmd = np.array(cmd)
        cmd = cmd * self.gain
        return cmd

    def display_query(self, report, observations, goal):  # @UnusedVariable
        ServoAgentInterface.display_query(self, report, observations, goal)
        
        f = report.figure(cols=4)
        
        y_goal = goal
        y0 = observations
        
        D = get_distance_map(y0, y_goal)
        
        f.data('D', D).display('scale').add_to(f, caption='Distance matrix')
        
        plans = self.find_best(y0, y_goal, limit=4)
        for p in plans:
            motion = self.plans[p]
            ygoal1, ygoal1u = motion.predict_inverse(y_goal)
            R = motion.get_R()
            
            DR = D.copy()
            DR[R > 0] = np.nan
    
            ff = report.figure()
            ff.data('DR', DR).display('scale').add_to(ff, caption='DR')
            
            known = ygoal1u > 0
            unknown = np.logical_not(known)
            
            sensels = np.array(range(y0.size))
            
            with ff.plot('prediction2') as pylab:
                pylab.plot(y0, 'b-', label='y0')
                pylab.plot(sensels[known], ygoal1[known], 'rs', label='ygoal1')
                pylab.plot(sensels[unknown], ygoal1[unknown], 'rx', label='ygoal1u')
                
                #from yc1 304.s10_servo_field.plots import plot_style_sensels
                #plot_style_sensels(pylab)   
                pylab.legend()
        



@contract(A='array[NxN]')
def myexp(A):
    """ A stable approximation of the exponential of a matrix. """
    warnings.warn('myexp(): to test')
    N = A.shape[0]
    res = np.zeros((N, N))
    for i in range(1, N - 1):
        x1 = A[i, i - 1]
        x2 = A[i, i + 1]  
        
        resp = x1 - x2
        j = i + resp
        j1 = np.ceil(j)
        j2 = np.floor(j)
        r2 = 1 - (j - j2)
        r1 = 1 - r2
        if 0 <= j1 < N:
            res[i, j1] = r1
        if 0 <= j2 < N:
            res[i, j2] = r2
    return res

@contract(y0='array[N]', y1='array[N]')
def get_distance_map(y0, y1):
    N = y0.size
    D = np.zeros((N, N))
    for i in range(N):
        D[i, :] = np.abs(y0[i] - y1)
    return D
    
    
