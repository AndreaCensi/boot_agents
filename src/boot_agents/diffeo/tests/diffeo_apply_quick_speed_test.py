from boot_agents.diffeo.diffeo_apply_quick import FastDiffeoApply
import numpy as np
from boot_agents.diffeo.library.testing import for_all_dd
from boot_agents.diffeo.diffeo_basic import diffeo_apply
from bootstrapping_olympics.utils.in_a_while import InAWhile


#@for_all_dd
def diffeo_quick1(id_dd, dd): #@UnusedVariable
    M, N = dd.shape[0], dd.shape[1]
    template = np.random.rand(M, N)

    fda = FastDiffeoApply(dd)
    
    functions = dict(regular=lambda: diffeo_apply(dd, template),
                     fast=lambda: fda(template))

    repeat = 100
    for name, function in functions.items():
        fps = InAWhile()
        for _ in range(repeat):
            fps.its_time()
            _ = function()
        print('%s: %s fps' % (name, fps.fps()))
