from . import contract, np

@contract(D='valid_diffeomorphism')
def diffeomorphism_to_rgb(D):
    ''' Displays a diffeomorphism as a 3D image. '''
    M, N = D.shape[0], D.shape[1]
    side = int(np.ceil(M / 10.0))
    rgb = np.zeros((M, N, 3), 'uint8')
    
    rgb[:, :, 0] = ((D[:, :, 0] / side) % 2) * 255
    #rgb[:, :, 1] = (D[:, :, 0] * 120) / M + (D[:, :, 1] * 120) / N
    rgb[:, :, 1] = 0 
    rgb[:, :, 2] = ((D[:, :, 1] / side) % 2) * 255

    return rgb
