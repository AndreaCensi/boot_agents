from conf_tools import ConfigMaster


__all__ = [
    'get_bgds_config',
    'get_conftools_bgds_models',
    'get_conftools_bgds_estimators',     
]

class BGDSConfig(ConfigMaster):
    def __init__(self):
        ConfigMaster.__init__(self, 'bgds')
        from .bgds_model import BGDSmodel
        self.add_class_generic('bgds_models', '*.bdse_models.yaml', 
                       BGDSmodel)
        from .bgds_estimator import BGDSEstimator
        self.add_class_generic('bgds_estimators', '*.bdse_estimators.yaml', 
                               BGDSEstimator)

get_bgds_config = BGDSConfig.get_singleton

def get_conftools_bgds_estimators():
    return get_bgds_config().bgds_estimators

def get_conftools_bgds_models():
    return get_bgds_config().bgds_models
