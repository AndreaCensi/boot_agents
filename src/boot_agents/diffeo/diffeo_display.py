from . import contract, np, diffeo_identity
from reprep import scale
import itertools

@contract(D='valid_diffeomorphism')
def diffeomorphism_to_rgb(D, nquads=15):
    ''' Displays a diffeomorphism as an RGB image. '''
    M, N = D.shape[0], D.shape[1]
    side = int(np.ceil(M * 1.0 / nquads))
    rgb = np.zeros((M, N, 3), 'uint8')
    
    rgb[:, :, 0] = ((D[:, :, 0] / side) % 2) * 255
    #rgb[:, :, 1] = (D[:, :, 0] * 120) / M + (D[:, :, 1] * 120) / N
    rgb[:, :, 1] = 0 
    rgb[:, :, 2] = ((D[:, :, 1] / side) % 2) * 255

    return rgb


@contract(D='valid_diffeomorphism')
def diffeomorphism_to_rgb_cont(D):
    M, N = D.shape[0], D.shape[1]
    n = 3 
    rgb = np.zeros((M, N, 3), 'uint8')    
    rgb[:, :, 0] = (D[:, :, 0] * n * 255) / M
    rgb[:, :, 1] = 100
    rgb[:, :, 2] = (D[:, :, 1] * n * 255) / N
    return rgb
    

@contract(D='valid_diffeomorphism')
def diffeo_to_rgb_norm(D):
    norm, angle, dx, dy = diffeo_to_delta(D)
    return scale(norm)

@contract(D='valid_diffeomorphism', returns='array[HxWx3](uint8)')
def diffeo_to_rgb_angle(D):
    norm, angle, dx, dy = diffeo_to_delta(D)
    return angle2rgb(angle)

# TODO: add -pi, pi in contract
@contract(angle='array[HxW]', returns='array[HxWx3](uint8)')
def angle2rgb(angle, nan_color=[0, 0, 0]):
    H, W = angle.shape
    hsv = np.zeros((H, W, 3))
    for i, j in itertools.product(range(H), range(W)):
        hsv[i, j, 0] = (angle[i, j] + np.pi) / (2 * np.pi)
    hsv[:, :, 1] = 1
    hsv[:, :, 2] = 1
    
    isnan = np.isnan(angle)
    for k in range(3):
        hsv[:, :, k][isnan] = nan_color[k] 
    
    from scikits.image.color import hsv2rgb #@UnresolvedImport
    rgb = (hsv2rgb(hsv) * 255).astype('uint8')
    return rgb

@contract(shape='tuple(int,int)', returns='array[HxWx3](uint8)')
def angle_legend(shape, center=2.0):
    a = np.zeros(shape)
    H, W = shape
    for i, j in itertools.product(range(H), range(W)):
        x = i - H / 2.0
        y = j - W / 2.0
        r = np.hypot(x, y)
        if r > center:
            a[i, j] = np.arctan2(y, x)
        else:
            a[i, j] = np.nan
    return angle2rgb(a)
        
        
def diffeo_to_delta(D):
    ''' Returns norm, angle representation. '''
    identity = diffeo_identity(D.shape[0:2])
    dx = (D - identity)[:, :, 0]
    dy = (D - identity)[:, :, 1]
    angle = np.arctan2(dy, dx) 
    norm = np.hypot(dx, dy)
    angle[norm == 0] = np.nan
    return norm, angle, dx, dy

@contract(D='valid_diffeomorphism')
def diffeo_stats(D):
    norm, angle, dx, dy = diffeo_to_delta(D)
    s = ''
    s += 'Maximum norm: %f\n' % norm.max()
    s += 'Mean norm: %f\n' % np.mean(norm)
    s += 'Mean  d0: %f\n' % np.mean(dx)
    s += 'Mean  d0: %f\n' % np.mean(dy)
    s += 'Mean |d0|: %f\n' % np.mean(np.abs(dx))
    s += 'Mean |d1|: %f\n' % np.mean(np.abs(dy))
    return s

@contract(D='valid_diffeomorphism')
def diffeo_to_rgb_inc(D):
    norm, angle, dx, dy = diffeo_to_delta(D)
    angle_int = ((angle + np.pi) * 255 / (np.pi * 2)).astype('int') 
    if norm.max() > 0:
        norm = norm * 255 / norm.max() 
    M, N = D.shape[0], D.shape[1]
    rgb = np.zeros((M, N, 3), 'uint8')
    rgb[:, :, 0] = angle_int
    rgb[:, :, 1] = norm
    rgb[:, :, 2] = 1 - angle_int
    return rgb
