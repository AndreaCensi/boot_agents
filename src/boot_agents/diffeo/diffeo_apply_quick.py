from contracts import contract
from boot_agents.diffeo.flattening import Flattening
from boot_agents.diffeo.diffeo_basic import diffeo_apply
import numpy as np


class FastDiffeoApply():
    """ 
        A class that precomputes structures to fast
        apply a discrete diffeomorphism. 
        
        
    """
    @contract(dd='valid_diffeomorphism,array[HxWx2]')
    def __init__(self, dd):
        self._dd = dd
        H, W = dd.shape[:2]
        self.shape = (H, W)
        self.f = Flattening.by_rows((H, W))
        indices = self.f.get_cell2index()
        assert indices.shape == (H, W)
        indices_phi = diffeo_apply(dd, indices)
        assert indices_phi.shape == (H, W)
        self.indices_phi_flat = self.f.rect2flat(indices_phi)
                
    @contract(values='array[HxW]', returns='array[HxW]')
    def apply2d(self, values):
        """ Applies the diffeomorphism to a 2D image. """
        if values.shape != self.shape:
            msg = 'Expected shape %s, got %s.' % (self.shape, values.shape)
            raise ValueError(msg)
        values_flat = self.f.rect2flat(values)
        result_flat = values_flat[self.indices_phi_flat]
        result = self.f.flat2rect(result_flat)
        assert result.shape == self.shape
        return result
    
    @contract(template='array[MxNx...]')
    def __call__(self, template):
        """ Applies the diffeomorphism to a generic signal 
            with more components. """
        if template.ndim > 3 or template.ndim < 1:
            msg = 'Not implmemented for shapes like %s.' % str(template.shape) 
            raise NotImplementedError(msg)
        if template.ndim == 2:
            return self.apply2d(template)
        if template.ndim == 3:
            result = np.empty_like(template)
            nsignals = template.shape[2]
            for i in range(nsignals):
                result[:, :, i] = self.apply2d(template[:, :, i])
            return result
        assert False
