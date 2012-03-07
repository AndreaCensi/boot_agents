from . import contract, np, Publisher

__all__ = ['SymbolsStatistics']


class SymbolsStatistics:
    ''' 
        Computes the first- and second-order statistics of a stream of 
        symbols.
    '''

    @contract(n='int,>1', window='int,>=1')
    def __init__(self, n, window=3):
        ''' 
            Initializes the structure with the number of symbols.
        
            :param n: Number of symbols. 
        '''
        self.n = int(n)
        self.num_samples = 0
        self.window = window
        self.histogram = np.zeros(n, 'float')
        self.transitions = np.zeros((window, n, n), 'float')

        # transitions[delta, b, a] = number of times from a to b 

        self.history = []
        # TODO: check overflow

    @contract(x='int', dt='float,>0')
    def update(self, x, dt=1.0):
        if x >= self.n:
            msg = 'Invalid value %d, expected [0,%d]' % (x, self.n - 1)
            raise ValueError(msg)

        self.histogram[x] += dt

        if len(self.history) == self.window:
            for i, y in enumerate(self.history):
                delta = self.window - i
                self.transitions[delta - 1, x, y] += dt

        self.num_samples += dt

        # Update history
        self.history.append(x)
        if len(self.history) > self.window:
            self.history.pop(0)

    @contract(pub=Publisher)
    def publish(self, pub, skim=3):
        if self.num_samples == 0:
            pub.text('warning',
                     'Cannot publish anything as I was never updated.')
            return

        pub.text('stats', 'Num samples: %s' % self.num_samples)

        with pub.plot('histogram') as pylab:
            h = self.histogram / self.num_samples
            pylab.plot(h, 'k.')
            pylab.xlabel('symbol')
            pylab.ylabel('frequency')

        for delta in range(1, self.window + 1):
            sec = pub.section('delta%d' % delta)
            raw = self.get_raw_matrix(delta)
            trans = self.get_transition_matrix(delta)
            sec.array_as_image('raw',
                               raw,
                               filter='scale',
                               filter_params={'skim': skim},
                               caption='raw values')

            sec.array_as_image('transitions',
                               trans,
                               filter='scale',
                               filter_params={'skim': skim},
                               caption='transitions')

            with sec.plot('returns', caption='Return to same value') as pl:
                pl.plot(trans.diagonal(), 'k.')
                pl.ylabel('return probability')
                pl.xlabel('symbol')

    @contract(delta='int,>=1')
    def get_raw_matrix(self, delta):
        K = delta - 1
        if K >= self.transitions.shape[0]:
            msg = 'Invalid delta value %d (window = %d)' % (delta, self.window)
            raise ValueError(msg)
        return self.transitions[K, :, :]

    @contract(delta='int,>=1')
    def get_transition_matrix(self, delta):
        K = delta - 1
        if K >= self.transitions.shape[0]:
            msg = 'Invalid delta value %d (window = %d)' % (delta, self.window)
            raise ValueError(msg)

        X = self.transitions[K, :, :]
        # X[delta, b, a] = number of times from a to b
        T = X.copy()
        for j in range(int(self.n)): # XXX
            n = self.histogram[j]
            if n > 0:
                T[:, j] = T[:, j] / n
        return T
