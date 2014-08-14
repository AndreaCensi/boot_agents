from boot_agents.utils import check_scale_score, scale_score, scale_score_scipy
from bootstrapping_olympics.utils import InAWhile, assert_allclose
import itertools
import numpy as np


algos = {
    #'_argsort': lambda x: np.argsort(x),
    'scipy': lambda x: scale_score_scipy(x),
#    'quick2': lambda x: scale_score2(x, 'quicksort'),
#    'merge2': lambda x: scale_score2(x, 'mergesort'),
#    'heapsort2': lambda x: scale_score2(x, 'heapsort'),
    'quick-quick': lambda x: scale_score(x, 'quicksort', 'quicksort'),
    'quick-merge': lambda x: scale_score(x, 'quicksort', 'mergesort'),
    'quick-heap': lambda x: scale_score(x, 'quicksort', 'heapsort'),
    'merge-quick': lambda x: scale_score(x, 'mergesort', 'quicksort'),
    'merge-merge': lambda x: scale_score(x, 'mergesort', 'mergesort'),
    'merge-heap': lambda x: scale_score(x, 'mergesort', 'heapsort'),
    'heap-quick': lambda x: scale_score(x, 'heapsort', 'quicksort'),
    'heap-merge': lambda x: scale_score(x, 'heapsort', 'mergesort'),
    'heap-heap': lambda x: scale_score(x, 'heapsort', 'heapsort'),
}

data = {
        'v5': np.random.rand(5).astype('float32'),
        'v10': np.random.rand(10, 10).astype('float32'),
        'v50': np.random.rand(50, 50).astype('float32'),
        'v100': np.random.rand(100, 100).astype('float32'),
}

def check_scale_score_variants_test():
    print('Original scale_score')
    v = data['v5']
    x0 = scale_score(v)
    check_scale_score(v, x0)
    
    for id_algo, algo in algos.items():
        print('Variant %s' % id_algo)
        x = algo(v)
        check_scale_score(v, x)
        assert_allclose(x, x0)
    
def measure_fps():
    n = 100
    for id_data, id_algo in itertools.product(data, algos):
        v = data[id_data]
        algo = algos[id_algo]
    
        count = InAWhile()
        for _ in xrange(n):
            algo(v)
            count.its_time()
        
        fps = count.fps()
        print('%s-%15s: %s' % (id_data, id_algo, fps))

if __name__ == '__main__':
    check_scale_score_variants_test()
    measure_fps()

