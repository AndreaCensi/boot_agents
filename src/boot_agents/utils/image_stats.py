from . import contract, MeanVariance, Publisher, generalized_gradient


__all__ = ['ImageStats']


class ImageStats:
    """ A class to compute all statistics of an image (2D float) stream. """
    
    def __init__(self):
        self.mv = MeanVariance()
        self.gmv = [MeanVariance(), MeanVariance()] 
        self.num_samples = 0
        self.last_y = None

    def get_num_samples(self):
        return self.num_samples
    
    @contract(y='array[HxW]')
    def update(self, y, dt=1.0):
        self.last_y = y.copy()
        gy = generalized_gradient(y)
        
        self.mv.update(y, dt)
        for i in range(2):
            self.gmv[i].update(gy[i, ...], dt)

        self.num_samples += dt


    @contract(pub=Publisher)
    def publish(self, pub):
        if self.num_samples == 0:
            pub.text('warning',
                     'Cannot publish anything as I was never updated.')
            return

        stats = "Shape: %s " % str(self.last_y.shape)
        stats += 'Num samples: %s' % self.num_samples
        pub.text('stats', stats)


        mean, stddev = self.mv.get_mean_stddev()         
        pub.array_as_image('mean', mean)
        pub.array_as_image('stddev', stddev)
        
        for i in range(2):
            sec = pub.section('grad%d' % i)
            mean, stddev = self.gmv[i].get_mean_stddev()
            sec.array_as_image('mean', mean)
            sec.array_as_image('stddev', stddev)
            
      



    
    
