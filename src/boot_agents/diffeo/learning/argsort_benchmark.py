#from boot_agents.utils.nonparametric import scale_score, scale_score2
#import itertools
#import numpy as np
#from bootstrapping_olympics.utils.in_a_while import InAWhile
#
#
#
#
#algos = {
#    '_argsort': lambda x: np.argsort(x),
#    'quick2': lambda x: scale_score2(x, 'quicksort'),
#    'merge2': lambda x: scale_score2(x, 'mergesort'),
#    'heapsort2': lambda x: scale_score2(x, 'heapsort'),
#    'quick-quick': lambda x: scale_score(x, 'quicksort', 'quicksort'),
#    'quick-merge': lambda x: scale_score(x, 'quicksort', 'mergesort'),
#    'quick-heap': lambda x: scale_score(x, 'quicksort', 'heapsort'),
#    'merge-quick': lambda x: scale_score(x, 'mergesort', 'quicksort'),
#    'merge-merge': lambda x: scale_score(x, 'mergesort', 'mergesort'),
#    'merge-heap': lambda x: scale_score(x, 'mergesort', 'heapsort'),
#    'heap-quick': lambda x: scale_score(x, 'heapsort', 'quicksort'),
#    'heap-merge': lambda x: scale_score(x, 'heapsort', 'mergesort'),
#    'heap-heap': lambda x: scale_score(x, 'heapsort', 'heapsort'),
#}
#
#data = {
#    'v10': np.random.rand(10, 10).astype('float32'),
#    'v50': np.random.rand(50, 50).astype('float32'),
#    'v100': np.random.rand(100, 100).astype('float32'),
#}
#
#n = 100
#
#for id_data, id_algo in itertools.product(data, algos):
#    v = data[id_data]
#    algo = algos[id_algo]
#
#    count = InAWhile()
#    for _ in xrange(n):
#        algo(v)
#        count.its_time()
#    
#    fps = count.fps()
#    print('%s-%15s: %s' % (id_data, id_algo, fps))
