__version__ = '0.1'

import logging

logging.basicConfig();
logger = logging.getLogger("boot_agents")
logger.setLevel(logging.DEBUG)


from .simple_stats import *
from .bds import *
