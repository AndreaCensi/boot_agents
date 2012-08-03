from .. import BDSEmodel
from . import np, contract
from bootstrapping_olympics.unittests.utils import fancy_test_decorator
from boot_agents.bdse.testing.simulate import BDSSimulator
from boot_agents.bdse.model.bdse_estimator import BDSEEstimator


@contract(n='int,>=1', k='int,>=1')
def get_bds_M_N(n, k):
    M = np.zeros((n, n, k))
    N = np.zeros((n, k))
    return M, N


@contract(n='int,>=1')
def bdse_ex_one_command_indip(n):
    k = n
    M, N = get_bds_M_N(n=n, k=k)
    for i in range(n):
        M[i, i, i] = 1
    return BDSEmodel(M=M, N=N)


def bdse_random(n, k):
    M, N = get_bds_M_N(n=n, k=k)
    M = np.random.randn(*M.shape)
    N = np.random.randn(*N.shape)
    return BDSEmodel(M=M, N=N)


@contract(returns='dict')
def bdse_examples():
    """ Returns some examples of BDSe systems. """
    examples = {}
    # works ok
    examples['rand21'] = dict(model=bdse_random(n=2, k=1),
                              desc="")

    examples['rand32'] = dict(model=bdse_random(n=3, k=2),
                              desc="")

    examples['indip1'] = dict(model=bdse_ex_one_command_indip(n=1),
                              desc="")
    examples['indip3'] = dict(model=bdse_ex_one_command_indip(n=3),
                              desc="")
    return examples

all_bdse_examples = bdse_examples()

for_all_bdse_examples = fancy_test_decorator(
        lister=lambda: list(all_bdse_examples.keys()),
        arguments=lambda eid: (eid, all_bdse_examples[eid]['model']),
        attributes=lambda eid: dict(bdse_model=eid))


@for_all_bdse_examples
def check_instantiation(mid, model):
    pass


@for_all_bdse_examples
def check_simulation(mid, model): #@UnusedVariable
    y0 = lambda: np.random.rand(model.get_y_shape())
    u_dist = lambda: np.random.rand(model.get_u_shape())
    simulator = BDSSimulator(model, y0, u_dist)
    for _ in simulator.get_simulation(10, 0.1):
        pass


@for_all_bdse_examples
def check_learning(mid, model): #@UnusedVariable
    y0 = lambda: np.random.rand(model.get_y_shape())
    u = lambda: np.random.rand(model.get_u_shape())
    simulator = BDSSimulator(model, y0, u)

    estimator = BDSEEstimator()
    nsteps = 1
    nstart = 50000
#    nsteps = 10
#    nstart = 5000
    dt = 0.1
    for _ in range(nstart):
        for  y, u, y_dot in simulator.get_simulation(nsteps, dt):
            #printm('u', u)
            estimator.update(y, u, y_dot)

    model2 = estimator.get_model()

    print("Model:\n%s" % model.description())

    print("Learned:\n%s" % model2.description())


