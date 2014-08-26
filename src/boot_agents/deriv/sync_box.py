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
                        y_name, 
                        u_name, 
                        out_name,
                       ):
    """ 
        Returns a black box that takes as input:
    
            y_name
            u_name
        
        and outputs
        
            out_name = dict(y=..., y_dot=..., u=...)
    """ 
    rs = series(
            SyncRep(master=y_name),
            #Info('after-sync-rep'),     
            Route([
              # pass u through
               ({u_name:u_name}, Identity(), {u_name:u_name}),
              # pass y through
              #({'y':'y'}, Identity(), {'y':'y'}),
              # derivative 
              ({y_name:y_name}, 
                WrapTMfromT(SampledDerivInst()), 
               {y_name:'y_y_dot'}),
            ]),
            CollectSignals([u_name, 'y_y_dot'], ignore_if_incomplete=True), 
            InstantaneousTF(lambda x: repack(x, u_name=u_name,
                                             out_name=out_name)),
        )

    return rs

def repack(value, u_name, out_name):
    check_isinstance(value, dict)
    expected = set([u_name, 'y_y_dot'])
    got = set(value.keys())
    if not expected == got:
        msg = 'Incomplete signal set: %s, expected %s.' % (got, expected)
        raise ValueError(msg)
    y_y_dot = value['y_y_dot']
    y, y_dot = y_y_dot
    u = value[u_name]
    res = dict(y=y, y_dot=y_dot, u=u)
    return (out_name, res)
    
    


