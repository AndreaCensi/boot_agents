from . import np


def y_axis_balanced(pylab, extra_space=0.1, show0=True):
    a = pylab.axis()
    y_max = a[3]
    y_min = a[2]
    D = np.max([np.abs(y_max), np.abs(y_min)])
    D *= (1 + extra_space)
    pylab.axis((a[0], a[1], -D, +D))
    if show0:
        pylab.plot([a[0], a[1]], [0, 0], 'k--') # TODO: zdepth


def y_axis_positive(pylab, extra_space=0.1, show0=True):
    a = pylab.axis()
    y_max = a[3]
    y_min = -y_max * extra_space
    y_max *= (1 + extra_space)
    pylab.axis((a[0], a[1], y_min, y_max))
    if show0:
        pylab.plot([a[0], a[1]], [0, 0], 'k--') # TODO: zdepth


def y_axis_extra_space(pylab, extra_space=0.1):
    a = pylab.axis()
    D = a[3] - a[2]
    extra = D * extra_space
    pylab.axis((a[0], a[1], a[2] - extra, a[3] + extra))
