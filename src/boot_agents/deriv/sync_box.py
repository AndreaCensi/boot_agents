from blocks import SimpleBlackBox, series
from blocks.library import (CollectSignals, Identity, InstantaneousTF, Route, 
    SampledDerivInst, SyncRep, WrapTMfromT)
from contracts import contract
from contracts.utils import check_isinstance
from blocks.library.simple.info import Info

__all__ = [
    'get_sync_deriv_box',
]

@contract(returns=SimpleBlackBox)
def get_sync_deriv_box(
#                        y_name='y', 
#                        u_name='u', 
#                        out_name='y_u',
                       ):
    """ 
        Returns a black box that takes as input:
    
            y
            u
        
        and outputs
        
            y_u = dict(y=..., u=...)
    """ 
    rs = series(
            SyncRep(master='y'),
            Info('after-sync-rep'),     
            Route([
              # pass u through
               ({'u':'u'}, Identity(), {'u':'u'}),
              # pass y through
              #({'y':'y'}, Identity(), {'y':'y'}),
              # derivative 
              ({'y':'y'}, 
                WrapTMfromT(SampledDerivInst()), 
               {'y':'y_y_dot'}),
            ]),
            CollectSignals(['u', 'y_y_dot'], ignore_if_incomplete=True), 
            InstantaneousTF(repack),
        )

    return rs

def repack(value):
    check_isinstance(value, dict)
    expected = set(['u', 'y_y_dot'])
    got = set(value.keys())
    if not expected == got:
        msg = 'Incomplete signal set: %s, expected %s.' % (got, expected)
        raise ValueError(msg)
    y_y_dot = value['y_y_dot']
    y, y_dot = y_y_dot
    u = value['u']
    res = dict(y=y, y_dot=y_dot, u=u)
    return ('y_u', res)
    
    


