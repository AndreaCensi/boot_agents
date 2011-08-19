from . import contract, np
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
    
