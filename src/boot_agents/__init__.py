__version__ = '1.2dev1'

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from .misc_utils import *
from .simple_stats import *
from .geometry import *
from .robustness import *
from .recursive import *
from .deriv import *
#from boot_agents_bdse import *
#from boot_agents_bgds import *

def jobs_comptests(context):
    from conf_tools import GlobalConfig
    GlobalConfig.global_load_dirs(['boot_agents.configs'])

    # unittests for boot olympics
    import bootstrapping_olympics.unittests

    # instance    
    from comptests import jobs_registrar
    from bootstrapping_olympics import get_boot_config
    jobs_registrar(context, get_boot_config())


