from contracts import contract, new_contract
import itertools
new_contract('valid_2d_shape', 'seq[2](>0)')

__all__ = ['coords_iterate']

@contract(size='valid_2d_shape')
def coords_iterate(size):
    for x in itertools.product(range(size[0]), range(size[1])):
        yield x

