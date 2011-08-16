from . import contract, np, coords_iterate

@contract(size='array[2]((int32|int64),>=1)')
def cmap(size):
    for k in [0, 1]:
        if size[k] % 2 == 0:
            size[k] = size[k] + 1
    r = np.zeros(shape=(size[0], size[1], 2), dtype='int32')
    for i, j in coords_iterate(size):
        r[i, j, 0] = i - (size[0] - 1) / 2
        r[i, j, 1] = j - (size[1] - 1) / 2
    return r
