from .bgds_agent import *
from .bgds_agent_robust import *



def jobs_comptests(context):
    from conf_tools import GlobalConfig
    config_dirs = [
        'bootstrapping_olympics.configs',
        'boot_agents_bgds.configs',
    ]
    GlobalConfig.global_load_dirs(config_dirs)

    # unittests for boot olympics
    import bootstrapping_olympics.unittests

    # instance    
    from comptests import jobs_registrar
    from bootstrapping_olympics import get_boot_config
    jobs_registrar(context, get_boot_config())
