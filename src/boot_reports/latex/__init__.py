""" 
    Some utils to include the results of simulations inside 
    a LaTeX document. 
"""
from .. import getLogger
logger = getLogger(__name__)


class Globals:
    sets_dir = None
    resources_dir = None
    
    
def get_sets_dir():
    if Globals.sets_dir is None:
        raise Exception('please initialize using set_sets_dir()')
    return Globals.sets_dir

    
def set_sets_dir(d):
    Globals.sets_dir = d
    
    
def set_resources_dir(d):
    Globals.resources_dir = d


def get_resources_dir():
    if Globals.resources_dir is None:
        raise Exception('please initialize using set_resources_dir()')
    return Globals.resources_dir
    
from .utils import *
from .names import *
from .load import *
from .vehicle import *

from .bds import *
from .prediction import *
from .servo_stat import *
from .templates import *
from .templates_table import *

