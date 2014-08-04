from blocks import SimpleBlackBox
from blocks.library import Identity, Route, SampledDerivInst, WrapTMfromT
from contracts import contract
from blocks.composition import series
from blocks.library.timed_named.collect import CollectSignals
from blocks.library.simple.instantaneous import InstantaneousTF
from blocks.library.timed.checks import check_timed_named, check_timed
from contracts.utils import check_isinstance

__all__ = [
    'get_sync_deriv_box',
]

@contract(returns=SimpleBlackBox)
def get_sync_deriv_box(y_name='y', 
                       u_name='u', 
                       out_name='y_u'):
    """ 
        Returns a black box that takes as input:
    
            y
            u
        
        and outputs
        
            y_u = dict(y=..., u=...)
    """ 
    deriv = WrapTMfromT(SampledDerivInst())
    
    r = Route([
          # pass u through
           ({'u':'u'}, Identity(), {'u':'u'}),
          # pass y through
          #({'y':'y'}, Identity(), {'y':'y'}),
          # derivative 
          ({'y':'y'}, deriv, {'y':'y_y_dot'}),
        ])
    
    rs = series(r, 
                CollectSignals(['u', 'y_y_dot'], error_if_incomplete=True), 
                InstantaneousTF(repack))

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
    
    


