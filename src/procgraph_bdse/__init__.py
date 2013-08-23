
procgraph_info = {
    # List of python packages 
    'requires': ['bootstrapping_olympics']
}

from .reports import *

from procgraph import pg_add_this_package_models
pg_add_this_package_models(__file__, __package__)
