__version__ = '1.0'

import logging

logging.basicConfig();
logger = logging.getLogger("BootAgents")
logger.setLevel(logging.DEBUG)

import numpy as np
from contracts import contract

from .simple_stats import *
from .geometry import *
from .bds import *
from .bgds import *
from .diffeo import *
