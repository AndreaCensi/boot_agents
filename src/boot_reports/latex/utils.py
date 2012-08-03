from . import logger
from StringIO import StringIO
from bootstrapping_olympics.utils import isodate
from reprep import Image_from_array
import numpy as np
import tempfile


def get_sensel_pgftable(V, what, desc='No description given'):
    s = StringIO()
    s.write('# Created by get_pgf_format()\n')
    s.write('# %s \n' % isodate())
    s.write('# sensel: counter of the sensel from 0 to n-1.\n')
    s.write('# %s: %s \n' % (what, desc))
    s.write('# senseln: counter of the sensel from 0 to 1.\n')
    s.write('sensel senseln %s\n' % what)    
    index = np.linspace(0, 1, V.size)
    for i, x in enumerate(V):
        s.write('%d %g %g\n' % (i, x, index[i]))
    return s.getvalue()


def jpg_data(rgb, quality=85, optimize=True):
    # http://mail.python.org/pipermail/image-sig/1999-August/000816.html
    import ImageFile
    ImageFile.MAXBLOCK = 1000000 # default is 64k
    im = Image_from_array(rgb)
    tmp = tempfile.NamedTemporaryFile(suffix='.jpg')
    name = tmp.name
    # TODO: try / redo
    try:
        im.save(name, quality=quality, optimize=optimize)
    except Exception as e:
        logger.error('Could not convert: %s' % e)
        im.save(name)
        
    return open(name, 'rb').read()


def png_data(rgb):
    im = Image_from_array(rgb)
    tmp = tempfile.NamedTemporaryFile(suffix='.png')
    name = tmp.name
    im.save(name)
    return open(name, 'rb').read()
