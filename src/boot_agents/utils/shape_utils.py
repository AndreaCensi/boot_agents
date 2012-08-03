
def expect_shape(name, vector, shape):
    if vector.shape != shape:
        msg = ('Expected shape %s for %r but found %s' % 
               (shape, name, vector.shape))
        raise ValueError(msg)
