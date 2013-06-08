from geometry.formatting import formatm

def expect_shape(name, vector, shape):
    if vector.shape != shape:
        msg = ('Expected shape %s for %r but found %s' % 
               (shape, name, vector.shape))
        if vector.size < 100:
            msg += '\n' + formatm(vector) 
        raise ValueError(msg)
