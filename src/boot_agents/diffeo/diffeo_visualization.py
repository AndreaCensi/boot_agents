from reprep.graphics.filter_scale import scale
import numpy as np

def scalaruncertainty2rgb(x, umin=0, umax=1):
    """ Converts the scalar uncertainty (in [min, max]) to rgb. (green=1, red=0) """
    # set exactly 0 to nan, and color it gray
    x = x.copy()
    
#    if min is None:
#        min = np.min(x)
#    if max is None:
#        max = np.nanmax(x)
    
    x[x == 0] = np.nan
    
    rgb = scale(x, max_value=umax, min_value=umin,
                   min_color=[1, 0, 0], max_color=[0, 1, 0],
                   nan_color=[0.5, 0.5, 0.5])
    return rgb
    
    

