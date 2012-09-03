from reprep.graphics.filter_scale import scale
import numpy as np

def scalaruncertainty2rgb(x):
    """ Converts the scalar uncertainty (in [0,1]) to rgb. (green=1, red=0) """
    # set exactly 0 to nan, and color it gray
    x = x.copy()
    x[x == 0] = np.nan
    
    rgb = scale(x, max_value=1, min_value=0,
                   min_color=[1, 0, 0], max_color=[0, 1, 0],
                   nan_color=[0.5, 0.5, 0.5])
    return rgb
    
    

