from contracts import contract, new_contract
import numpy as np
from reprep import Report
import itertools
import warnings
from .interface import BDSEServoInterface

__all__ = ['ServoLongTermMotion', 'BDSEServoLongTerm', 'myexp']


class ServoLongTermMotion():
    
    @contract(R='array[NxN](>=0,<=1)')
    def __init__(self, R):
        self.R = R.astype('float32')
        self.Rw = np.sum(self.R, axis=1)
        
    def get_R(self):
        return self.R
    
    @contract(returns='tuple(array[N], array[N](>=0,<=1))')
    def predict_inverse(self, y1):
        y1 = y1.astype('float32')
        y0 = np.dot(self.R, y1)
        return y0, self.Rw
    
    def predict(self, y0):
        raise NotImplemented()
        
new_contract('ServoLongTermMotion', ServoLongTermMotion)
        
        
class BDSEServoLongTerm(BDSEServoInterface):
    """ This uses long-term prediction. """
    
    # @contract(plans='dict(tuple(tuple, float):ServoLongTermMotion)')
    def __init__(self, grid, gain=0.045):
        self.plans = None
        self.bdse_model = None
        self.grid = grid
        self.gain = gain
    
    
    def init(self, boot_spec):
        self.boot_spec = boot_spec
        self.commands_spec = boot_spec.get_commands()
        
        
    def set_model(self, model):
        M = model.M
        K = M.shape[2]
        cmds = BDSEServoLongTerm.get_cmds(K=K, **self.grid)
    
        plans = {}
        for plan in cmds:
            cmd, t = plan
            u = np.array(cmd) * t
            A = np.einsum("abc,c -> ab", M, u)
            R = myexp(A)
            motion = ServoLongTermMotion(R)
            plans[(plan)] = motion
        
        self.plans = plans
        
    @staticmethod
    def get_cmds(K, mode, grid_max, grid_n):
        assert mode == 'uniform'
        cmds = []
        base = np.linspace(-grid_max, grid_max, grid_n)

        xs = itertools.product(base, repeat=K)

        for x in xs:
            x = np.array(x)
            t = np.linalg.norm(x)
            x = x / t
            cmds.append((tuple(x), t))
        return cmds
    
    def report(self):
        r = Report('ServoLongTerm')
        f = r.figure()
        for plan, result in self.plans.items():
            # cmd, time = plan
            what = str(plan)
            f.data(what, result > 0).display('scale').add_to(f, caption=what)
        return r
        
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
        print sols2
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
    
    
