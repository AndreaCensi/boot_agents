from conf_tools import ConfigMaster


__all__ = [
    'get_bdse_config',
    'get_conftools_bdse_models',
    'get_conftools_bdse_estimators',     
]

class BDSEConfig(ConfigMaster):
    def __init__(self):
        ConfigMaster.__init__(self, 'bdse')
        from bdse import BDSEmodel, BDSEEstimator
        self.add_class_generic('bdse_models', '*.bdse_models.yaml', 
                       BDSEmodel)
        self.add_class_generic('bdse_estimators', '*.bdse_estimators.yaml', 
                               BDSEEstimator)

get_bdse_config = BDSEConfig.get_singleton

def get_conftools_bdse_estimators():
    return get_bdse_config().bdse_estimators

def get_conftools_bdse_models():
    return get_bdse_config().bdse_models
