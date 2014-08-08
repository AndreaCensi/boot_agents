from .generation import for_all_bdse_models, for_all_bdse_models_estimators
from .simulate import BDSSimulator
from bdse import BDSEEstimator, BDSEmodel
from contracts.utils import check_isinstance
import numpy as np

@for_all_bdse_models
def check_instantiation(id_model, model):  # @UnusedVariable
    check_isinstance(model, BDSEmodel)

@for_all_bdse_models
def check_simulation(id_model, model):  # @UnusedVariable
    y0 = lambda: np.random.rand(model.get_y_shape())
    u_dist = lambda: np.random.rand(model.get_u_shape())
    simulator = BDSSimulator(model, y0, u_dist)
    for _ in simulator.get_simulation(10, 0.1):
        pass


@for_all_bdse_models_estimators
def check_learning(id_model, model, id_estimator, estimator):
    check_isinstance(model, BDSEmodel)
    check_isinstance(estimator, BDSEEstimator)
    print(id_model, id_estimator)
    if False:
        # Everything zero mean
        y0 = lambda: np.random.randn(model.get_y_shape())
        u = lambda: np.random.randn(model.get_u_shape())
    elif False:
        y0 = lambda: np.random.rand(model.get_y_shape())
        u = lambda: 5 * np.random.randn(model.get_u_shape())
    else:
        # everything in [0,1]
        y0 = lambda: 20 * np.random.rand(model.get_y_shape())
        u = lambda: 5 * np.random.rand(model.get_u_shape())
    simulator = BDSSimulator(model, y0, u)

    #    nsteps = 1
    #    nstart = 50000
    nsteps = 1
    nstart = 50000
    #    nstart = 5000000
    dt = 0.1
    count = 0
    for _ in range(nstart):
        for  y, u, y_dot in simulator.get_simulation(nsteps, dt):
            # printm('u', u)
            estimator.update(y, u, y_dot)
            count += 1
            if count % 500 == 0: 
                model2 = estimator.get_model()
    
                print("Model:\n%s" % model.description())
                print("Learned:\n%s" % model2.description())


