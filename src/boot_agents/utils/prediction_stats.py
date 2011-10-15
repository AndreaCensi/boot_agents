from . import Expectation, np, contract, MeanVariance, Publisher

class PredictionStats: 
    
    @contract(label_a='str', label_b='str')
    def __init__(self, label_a='a', label_b='b'):
        self.label_a = label_a
        self.label_b = label_b
        self.Ea = MeanVariance()
        self.Eb = MeanVariance()
        self.Edadb = Expectation()
        self.R = None
        self.R_needs_update = True
        self.num_samples = 0
        self.last_a = None
        self.last_b = None
        
    @contract(a='array[K]', b='array[K]', dt='float,>0')
    def update(self, a, b, dt=1.0):
        self.Ea.update(a, dt)
        self.Eb.update(b, dt)
        da = a - self.Ea.get_mean()
        db = b - self.Eb.get_mean()
        self.Edadb.update(da * db, dt)
        self.num_samples += dt
        
        self.R_needs_update = True
        self.last_a = a
        self.last_b = b
        
    def get_correlation(self):
        ''' Returns the correlation between the two streams. '''
        if self.R_needs_update:
            std_a = self.Ea.get_std_dev()
            std_b = self.Eb.get_std_dev()
            p = std_a * std_b
            zeros = p == 0
            p[zeros] = 1
            R = self.Edadb() / p
            R[zeros] = np.NAN 
            self.R = R
        self.R_needs_update = False
        return self.R
        

    @contract(pub=Publisher)    
    def publish(self, pub):
        if self.num_samples == 0:
            pub.text('warning',
                     'Cannot publish anything as I was never updated.')
            return

        pub.text('stats', 'Num samples: %s' % self.num_samples)

        R = self.get_correlation()
        
        with pub.plot('correlation') as pylab:
            pylab.plot(R, 'k.')
            pylab.axis((0, R.size, -1.1, +1.1))
            
        with pub.plot('last') as pylab:
            pylab.plot(self.last_a, 'g.', label=self.label_a)
            pylab.plot(self.last_b, 'm.', label=self.label_b)
            a = pylab.axis()
            m = 0.1 * (a[3] - a[2])
            pylab.axis((a[0], a[1], a[2] - m, a[3] + m))
            pylab.legend()
            
        with pub.plot('vs') as pylab:
            pylab.plot(self.last_a, self.last_b, '.')
            pylab.xlabel(self.label_a)
            pylab.ylabel(self.label_b)
            pylab.axis('equal')
            
        self.Ea.publish(pub.section('%s_stats' % self.label_a))
        self.Eb.publish(pub.section('%s_stats' % self.label_b))
            
