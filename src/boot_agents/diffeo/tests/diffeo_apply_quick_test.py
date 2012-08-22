from boot_agents.diffeo.diffeo_apply_quick import FastDiffeoApply
from boot_agents.diffeo.diffeo_basic import diffeo_apply
from boot_agents.diffeo.library.testing import for_all_dd
from geometry.utils.numpy_backport import assert_allclose
import numpy as np


@for_all_dd
def diffeo_quick1(id_dd, dd): #@UnusedVariable
    M, N = dd.shape[0], dd.shape[1]

    template = np.random.rand(M, N)
    y1 = diffeo_apply(dd, template)
    
    fda = FastDiffeoApply(dd)
    y2 = fda(template)
    
    assert_allclose(y1, y2)


@for_all_dd
def diffeo_quick2(id_dd, dd): #@UnusedVariable
    M, N = dd.shape[0], dd.shape[1]

    template = np.random.rand(M, N, 3)
    y1 = diffeo_apply(dd, template)
    
    fda = FastDiffeoApply(dd)
    y2 = fda(template)
    
    assert_allclose(y1, y2)
