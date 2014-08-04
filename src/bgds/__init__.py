from .utils import *
from .bgds_model import *
from .bgds_system import *
from .bgds_estimator import *
from .bgds_predictor import *
from .bgds_estimator_robust import *


from .configuration import *


def jobs_comptests(context):
    from conf_tools import GlobalConfig
    GlobalConfig.global_load_dirs(['bgds.configs'])

    # unittests for boot olympics
    from . import unittests

    # instance    
    from comptests import jobs_registrar
    jobs_registrar(context, get_bgds_config())

