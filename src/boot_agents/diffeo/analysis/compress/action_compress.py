from ... import diffeo_norm_L2
from ..action import Action
from geometry import printm # FIXME: dependency
import itertools
import numpy as np


def actions_compress(actions, threshold):
    ''' Compresses actions, checking if there are opposite commands pairs. 
    
        Returns a tuple new_actions, dict with other info.
    '''
    n = len(actions)
    M = np.zeros((n, n))
    print('Compressing %d actions.' % n)
    for i, j in itertools.product(range(n), range(n)):
        a1 = actions[i]
        a2 = actions[j]
        sim = Action.similarity(a1, a2)
        M[i, j] = sim
        print('- %s %s %s' % (a1, a2, sim))
#    baseline = np.abs(M).min()
    printm('M', M)
#    m0 = M / M.mean()

    for i in range(n):
        print('action[%d] = %s' % (i, actions[i]))

    Distance = np.zeros((n, n))
    for i, j in itertools.product(range(n), range(n)):
        Distance[i, j] = Action.distance(actions[i], actions[j])

    scale = Distance.mean()
    Distance = Distance / scale
    printm('Distance', Distance)

    Distance_to_inverse = np.zeros((n, n))
    for i, j in itertools.product(range(n), range(n)):
        Distance_to_inverse[i, j] = Action.distance_to_inverse(actions[i],
                                                               actions[j])
    Distance_to_inverse = Distance_to_inverse / scale
    printm('DistToInv', Distance_to_inverse)

#    print('Baseline similarity is %g' % baseline)
    # subtract, normalize
#    M2 = np.abs(M) - baseline
#    M2 = M2 / np.max(M2)
#    M2 = M2 * np.sign(M)
#    printm('M2', M2)
    # set diagonal to zero
    M2d = M.copy()
    np.fill_diagonal(M2d, 0)
    necessary = set()
    redundant = set()

    # first remove too similar actions
    for i in range(n):
        if i in redundant:
            continue
        print('* Cmd %s seems necessary.' % actions[i])
        necessary.add(i)
        # look for most similar commands
        while True:
            most = np.argmax(M2d[i, :])
            if M2d[i, most] > threshold:
                # match it as same command
                # (just remove it)
                print(' - removing %s because too sim  (%s)' %
                      (actions[most], M2d[i, most]))
                redundant.add(most)
                M2d[:, most] = 0
                M2d[most, :] = 0
            else:
                print(' - no more; max is %s at %s' % (actions[most],
                                                       M2d[i, most]))
                break
        # look for opposite
        while True:
            most = np.argmin(M2d[i, :])
            if M2d[i, most] < -threshold:
                # match it as same command
                # (just remove it)
                print(' - remove %s because opposite (%s)' %
                      (actions[most], M2d[i, most]))
                redundant.add(most)
                M2d[:, most] = 0
                M2d[most, :] = 0
                actions[i].invertible = True
            else:
                print(' - no more; min is %s at %s' % (actions[most],
                                                       M2d[i, most]))
                break

    compressed = []
    for i in range(n):
        if i in necessary:
            compressed.append(actions[i])
    info = dict()

    info['Distance'] = Distance
    info['Distance_to_inverse'] = Distance_to_inverse

    return (compressed, info)


def actions_commutators(actions):
    n = len(actions)
    actions2 = list(actions)
    for i, j in itertools.product(range(n), range(n)):
        if i <= j: continue
        if actions[i].invertible and actions[j].invertible:
            print('* Creating commutator of %s and %s' % (actions[i], actions[j]))
            actions2.append(Action.commutator(actions[i], actions[j]))
    return actions2

def actions_remove_similar_to_identity(actions, threshold):
    actions2 = []
    for action in actions:
        d = diffeo_norm_L2(action.d)
        print('* %20s from id: %s' % (action, d))
        if d < threshold:
            print(' removing because too close')
        else:
            actions2.append(action)
    return actions2





