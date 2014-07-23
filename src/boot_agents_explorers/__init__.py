from .exp_switcher import *
from .canonical_commands_agents import *


def jobs_comptests(context):
    from pkg_resources import resource_filename  # @UnresolvedImport
    dirname = resource_filename("boot_agents_explorers", "configs")

    from bootstrapping_olympics import get_boot_config
    from comptests import jobs_registrar

    boot_config = get_boot_config()
    boot_config.load(dirname)

    # unittests for boot olympics
    import bootstrapping_olympics.unittests
    j1 = jobs_registrar(context, boot_config)

    return j1

