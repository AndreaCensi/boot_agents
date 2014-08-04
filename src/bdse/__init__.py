from .model import *
from .estimators import *
from .configuration import *


def jobs_comptests(context):
    from conf_tools import GlobalConfig
    GlobalConfig.global_load_dirs(['bdse.configs'])

    # unittests for boot olympics
    from . import unittests

    # instance    
    from comptests import jobs_registrar
    jobs_registrar(context, get_bdse_config())

