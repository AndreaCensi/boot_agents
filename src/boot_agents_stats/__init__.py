from .cmd_stats import *
from .est_distribution import *
from .est_stats_2d import *
from .est_stats import *
from .est_stats_th import *
from .symbols_stats import *


def jobs_comptests(context):
    from conf_tools import GlobalConfig
    GlobalConfig.global_load_dirs(['boot_agents_stats.configs'])

    # unittests for boot olympics
    import bootstrapping_olympics.unittests
    
    from comptests import jobs_registrar
    from bootstrapping_olympics import get_boot_config
    jobs_registrar(context, get_boot_config())


