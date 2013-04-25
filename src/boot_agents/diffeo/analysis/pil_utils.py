from PIL import Image  # @PydevCodeAnalysisIgnore 
import numpy as np


def imread(filename):
    ''' 
        Reads an image from a file.
        
        :param filename: Image filename.
        :type filename: string
        
        :return: image: The image as a numpy array.
        :rtype: image
    '''
    try:
        im = Image.open(filename)
    except Exception as e:
        raise Exception('Could not open filename "%s": %s' % 
                        (filename, e))

    data = np.array(im)

    return data


def resize(value, width=None, height=None, mode=Image.NEAREST):
    ''' 
        Resizes an image.
        
        You should pass at least one of ``width`` or ``height``.
        
        :param value: The image to resize.
        :type value: rgb
        
        :param width: Target image width.
        :type width: int,>0
        
        :param height: Target image height.
        :type height: int,>0

        :return: image: The image as a numpy array.
        :rtype: rgb
    '''

    image = Image_from_array(value)

    if width is None and height is None:
        raise ValueError('You should pass at least one of width and height.')

    if width is None and height is not None:
        width = (height * image.size[0]) / image.size[1]
    elif height is None and width is not None:
        height = (width * image.size[1]) / image.size[0]

    # TODO: RGBA?
    image = image.resize((width, height), mode)
    return np.asarray(image.convert("RGB"))


def Image_from_array(a):
    ''' Converts an image in a numpy array to an Image instance.
        Accepts:  h x w      255  interpreted as grayscale
        Accepts:  h x w x 3  255  rgb  
        Accepts:  h x w x 4  255  rgba '''

    if not a.dtype == 'uint8':
        raise ValueError('I expect dtype to be uint8, got "%s".' % a.dtype)

    if len(a.shape) == 2:
        height, width = a.shape
        rgba = np.zeros((height, width, 4), dtype='uint8')
        rgba[:, :, 0] = a
        rgba[:, :, 1] = a
        rgba[:, :, 2] = a
        rgba[:, :, 3] = 255
    elif len(a.shape) == 3:
        height, width = a.shape[0:2]
        depth = a.shape[2]
        rgba = np.zeros((height, width, 4), dtype='uint8')
        if not depth in [3, 4]:
            raise ValueError('Unexpected shape "%s".' % str(a.shape))
        rgba[:, :, 0:depth] = a[:, :, 0:depth]
        if depth == 3:
            rgba[:, :, 3] = 255
    else:
        raise ValueError('Unexpected shape "%s".' % str(a.shape))

    im = Image.frombuffer("RGBA", (width, height), rgba.data,
                           "raw", "RGBA", 0, 1)
    return im
