from boot_agents.utils import ExpectationSlow, ExpectationFast
from numpy.testing.utils import assert_allclose
import numpy as np

def test_efficient_exp():
    shape = (4, 4)
    for exp_class in [ExpectationSlow, ExpectationFast]:
        yield check_expectation_one, exp_class

        sequence = (np.random.randn(*shape) for i in range(1))
        yield check_efficient_exp, exp_class, sequence

        sequence = (np.random.randn(*shape) for i in range(5))
        yield check_efficient_exp, exp_class, sequence
    
def check_expectation_one(exp_class):
    e = exp_class()
    x = np.random.rand(2, 2)
    dt = np.random.rand()
    e.update(x, dt)
    v = e.get_value()
    assert_allclose(v, x)
    
def check_efficient_exp(exp_class, sequence):
    exp1 = exp_class()
    xs = [] 
    T = 0
    for x in sequence:
        dt = np.random.rand()
        xs.append(x * dt)
        T += dt 
        exp1.update(x, dt=dt)
        
    es1 = exp1.get_value()
    expected = np.array(xs).sum(axis=0) / T
    
    assert_allclose(es1, expected)
