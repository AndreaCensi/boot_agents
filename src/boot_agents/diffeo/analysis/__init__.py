import logging

logging.basicConfig() # XXX
logger = logging.getLogger("diffeo_analysis")
logger.setLevel(logging.DEBUG)

from .pil_utils import *

from .plan import *
from .compress import *
