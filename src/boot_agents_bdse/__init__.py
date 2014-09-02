from .bds_robot import *
from .bdse_agent import *
from .bdse_agent2 import *
from .bdse_agent_robust import *
from .bdse_predictor import *
from .servo import *


def jobs_comptests(context):
    from conf_tools import GlobalConfig
    
    config_dirs = [
        'bootstrapping_olympics.configs',
        'boot_agents_bdse.configs',
        'bdse.configs',
    ]
    GlobalConfig.global_load_dirs(config_dirs)

    # unittests for boot olympics
    import bootstrapping_olympics.unittests

    # instance    
    from comptests import jobs_registrar
    from bootstrapping_olympics import get_boot_config
    jobs_registrar(context, get_boot_config())

