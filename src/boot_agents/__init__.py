__version__ = '1.1'

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from .misc_utils import *
from .simple_stats import *
from .geometry import *
from .robustness import *

from .bdse import *
from .bgds import *
from .bgds_agents import *


def get_comptests():
    # get testing configuration directory 
    from pkg_resources import resource_filename  # @UnresolvedImport
    dirname = resource_filename("boot_agents", "configs")
    
    # load into bootstrapping_olympics
    from comptests import get_comptests_app
    from bootstrapping_olympics import get_boot_config
    boot_config = get_boot_config()
    boot_config.load(dirname)
    
    # Our tests are its tests with our configuration
    import bootstrapping_olympics
    return bootstrapping_olympics.get_comptests()
