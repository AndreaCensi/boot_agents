__version__ = '1.1'

import logging

logging.basicConfig()
from logging import getLogger
logger = getLogger(__name__)
logger.setLevel(logging.DEBUG)

import numpy as np
from contracts import contract

from .misc_utils import *
from .simple_stats import *
from .geometry import *
from .robustness import *
from .bds import *
from .bdse import *
from .bgds import *
from .bgds_agents import *
from .diffeo import *

