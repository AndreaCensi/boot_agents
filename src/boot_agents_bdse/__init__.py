from .bds_robot import *
from .bdse_agent import *
from .bdse_agent2 import *
from .bdse_agent_robust import *
from .bdse_predictor import *
from .servo import *


def jobs_comptests(context):
    from conf_tools import GlobalConfig
    GlobalConfig.global_load_dirs(['boot_agents_bdse.configs'])

    # unittests for boot olympics
    import bootstrapping_olympics.unittests

    # instance    
    from comptests import jobs_registrar
    from bootstrapping_olympics import get_boot_config
    jobs_registrar(context, get_boot_config())

