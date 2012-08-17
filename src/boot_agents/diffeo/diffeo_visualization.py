from reprep.graphics.filter_scale import scale

def scalaruncertainty2rgb(x):
    """ Converts the scalar uncertainty (in [0,1]) to rgb. (green=1, red=0) """
    return scale(x, max_value=1, min_value=0,
                 min_color=[1, 0, 0], max_color=[0, 1, 0])

