from . import np
from contracts import contract
import itertools


#def scale_score(x, kind='quicksort'):
#    y = x.copy()
#    order = np.argsort(x.flat, kind=kind)
#    # Black magic ;-) Probably the smartest thing I came up with today. 
#    order_order = np.argsort(order, kind=kind)
#    y.flat[:] = order_order.astype(y.dtype)
#    return y

def scale_score(x, kind='quicksort', kind2='quicksort'):
    y = x.copy()
    order = np.argsort(x.flat, kind=kind)
    # Black magic ;-) Probably the smartest thing I came up with today. 
    order_order = np.argsort(order, kind=kind2)
    y.flat[:] = order_order.astype(y.dtype)
    return y

#@contract(x='array,shape(x)', returns='array(int32),shape(x)')
#def scale_score2(x, kind='quicksort', out=None):
#    if scale_score2.buf.size != x.size:
#        scale_score2.buf = np.arange(x.size, dtype='int32')
#    #r = np.arange(n, dtype='int32')
#    r = scale_score2.buf
##    print('x', x)
#    order = np.argsort(x.flat, kind=kind)
##    print('order', order)
#    order_order = r.take(order)
##    print('order_order', order_order)
#    return order_order.reshape(x.shape)
#scale_score2.buf = np.zeros(1)

def scale_score_scipy(x):
    import scipy, scipy.stats
    return scipy.stats.mstats.rankdata(x) - 1
    
#@contract(x='array,shape(x)', score='array(int32),shape(x)')
def check_scale_score(x, score):
    n = x.size
    xf = x.flat
    sf = score.flat
    for i, j in itertools.product(range(n), range(n)):
        if xf[i] > xf[j]:
            expect = sf[i] > sf[j]
        elif xf[i] < xf[j]:
            expect = sf[i] < sf[j]
        else:
            expect = sf[i] == sf[j]
            
        if not expect:
            msg = 'Found breach\n'
            msg += 'x     = %s\n' % x
            msg += 'score = %s\n' % score
            msg += 'i %s j %s \n' % (i, j) 
            msg += 'x[i] %s s[i] %s \n' % (xf[i], sf[i]) 
            msg += 'x[j] %s s[j] %s \n' % (xf[j], sf[j])
            raise ValueError(msg)
            
            
 
