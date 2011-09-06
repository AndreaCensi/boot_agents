__version__ = '0.1'

import logging

logging.basicConfig();
logger = logging.getLogger("boot_agents")
logger.setLevel(logging.DEBUG)

import numpy as np
from contracts import contract

from .simple_stats import *
from .geometry import *
from .bds import *
from .bgds import *
from .diffeo import *
