from .exp_switcher import *
from .canonical_commands_agents import *


def jobs_comptests(context):
    from conf_tools import GlobalConfig
    GlobalConfig.global_load_dirs(['boot_agents_explorers.configs'])

    # unittests for boot olympics
    import bootstrapping_olympics.unittests
    
    from comptests import jobs_registrar
    from bootstrapping_olympics import get_boot_config
    jobs_registrar(context, get_boot_config())

