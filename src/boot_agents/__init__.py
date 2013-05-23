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

