try:
    import procgraph
except ImportError:
    pass
else:
    from .pgblock import *
    from .pure_cmds import *
    from .to_dp_log import *
    
    from procgraph import pg_add_this_package_models
    pg_add_this_package_models(__file__, __package__)
