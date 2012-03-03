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
    def publish(self, pub):
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

        for i in range(self.window):
            trans = self.transitions[i, :, :]
            delta = i + 1
            pub.array_as_image('tran%d' % i,
                               trans,
                               filter='scale',
                               caption='delta=%d' % delta)

        for i in range(self.window):
            trans = self.get_transition_matrix(i)
            delta = i + 1
            pub.array_as_image('ntran%d' % i,
                               trans,
                               filter='scale',
                               caption='delta=%d' % delta)

    def get_transition_matrix(self, delta):
        X = self.transitions[delta, :, :]
        # X[delta, b, a] = number of times from a to b
        T = X.copy()
        for j in range(int(self.n)): # XXX
            n = self.histogram[j]
            if n > 0:
                T[:, j] = T[:, j] / n
        return T
