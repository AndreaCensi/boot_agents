from . import np


def scale_score(x):
    y = x.copy()
    order = np.argsort(x.flat)
    # Black magic ;-) Probably the smartest thing I came up with today. 
    order_order = np.argsort(order)
    y.flat[:] = order_order.astype(y.dtype)
    return y

