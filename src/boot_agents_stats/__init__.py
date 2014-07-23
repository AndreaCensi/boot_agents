from .cmd_stats import *
from .est_distribution import *
from .est_stats_2d import *
from .est_stats import *
from .est_stats_th import *
from .symbols_stats import *


def jobs_comptests(context):
    from pkg_resources import resource_filename  # @UnresolvedImport
    dirname = resource_filename("boot_agents_stats", "configs")

    from bootstrapping_olympics import get_boot_config
    from comptests import jobs_registrar

    boot_config = get_boot_config()
    boot_config.load(dirname)

    # unittests for boot olympics
    import bootstrapping_olympics.unittests
    j1 = jobs_registrar(context, boot_config)

    return j1

