import logging
getLogger = logging.getLogger
logger = getLogger(__name__)
logger.setLevel(logging.DEBUG) 
from .latex import *
from .notation import *
from .exp1208gl import *
