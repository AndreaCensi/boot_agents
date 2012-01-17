from boot_agents.diffeo import (Diffeomorphism2D, diffeo_inverse,
    diffeo_distance_L2, diffeo_compose)
from contracts import contract
import numpy as np


class Action:
    @contract(diffeo=Diffeomorphism2D)
    def __init__(self, diffeo, label, primitive, invertible, original_cmd):
        self.diffeo = diffeo
        self.label = label
        self.primitive = primitive
        self.invertible = invertible
        self.d = diffeo.d
        self.d_inv = diffeo_inverse(self.d)
        self.original_cmd = original_cmd

    def __str__(self):
        s = self.label
        if self.primitive:
            s += ' (%s)' % self.original_cmd
        return s

    @staticmethod
    def similarity(a1, a2):
        ''' Returns a similarity score between two actions
            1=same
            0=uncorrelated
            -1=inverse
        '''
        # These are distances, in [0, max]
        d_max = np.sqrt(2) / 2
        same_d = diffeo_distance_L2(a1.d, a2.d)
        oppo_d = 0.5 * (diffeo_distance_L2(a1.d, a2.d_inv) +
                        diffeo_distance_L2(a2.d, a1.d_inv))
        # we have to normalize [0,max] -> [1,0]
        # we go through an "angle" representation
        same_d = same_d / d_max * (np.pi / 2)
        oppo_d = oppo_d / d_max * (np.pi / 2)
        if same_d < oppo_d: # more similar than inverse
            return np.cos(same_d)
        else:
            return -np.cos(oppo_d)
#        

    @staticmethod
    def distance(a1, a2):
        return diffeo_distance_L2(a1.d, a2.d)

    @staticmethod
    def distance_to_inverse(a1, a2):
        oppo_d = 0.5 * (diffeo_distance_L2(a1.d, a2.d_inv) +
                        diffeo_distance_L2(a2.d, a1.d_inv))
        return oppo_d

    @staticmethod
    def commutator(a1, a2):
        # TODO: use uncertainty
        d1 = a1.d
        d1_inv = a1.d_inv
        d2 = a2.d
        d2_inv = a2.d_inv
        C = diffeo_compose
        z = C(C(C(d1, d2), d1_inv), d2_inv)
        diffeo = Diffeomorphism2D(z)
        label = "[%s,%s]" % (a1.label, a2.label)
        primitive = False
        invertible = True
        original_cmd = None
        return Action(diffeo, label, primitive, invertible, original_cmd)
