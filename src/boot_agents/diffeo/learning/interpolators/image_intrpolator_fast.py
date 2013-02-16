'''
Created on Feb 12, 2013

@author: adam
'''
from PIL import Image, ImageChops #@UnresolvedImport
import numpy as np
import pdb
from fractions import gcd
import itertools
import urllib
import cStringIO
from . import Interpolator


class ImageInterpolatorFast():
    def __init__(self, method):
        self.method = method
        
    def set_resolution(self, search_grid, area_size):
        self.search_grid = np.array(search_grid)
        self.area_size = np.array(area_size)
  
#    def set_image(self, image):
#        self.active_image = image
        
    def get_local_coord(self, index):
#        if index in [420, 421, 422]:
#            pdb.set_trace()
        res = self.search_grid
        XY = list(itertools.product(np.linspace(0, self.area_size[1] - 1, res[1]),
                                    np.linspace(0, self.area_size[0] - 1, res[0])))
        
#        steps = self._steps(self.area_size, res)
#        coord_offs = np.array(XY[index]) % steps
        
        local_coord = np.array(XY[index]) - self.area_size / 2 
        print('image_interpolator returns local_coord = ' + str(local_coord))
        return local_coord
#        return Interpolator().get_local_coord(self.area_size, self.search_grid, index)
        
    def reshape_image(self, array):
        area_size = self.area_size
        border_size = self.area_size / 2
        array_shape = np.array(array.shape)
        steps = self._steps(area_size, self.search_grid)
        self.arrays = {}
        img = Image.fromarray(array.astype('float'))
        border_img = self._add_border_pil(img, border_size)
        border_array = np.array(border_img)
#        print('  border image size is ' + str(border_img.size))
#        new_shape = border_array.shape * np.array(self.search_grid) / np.array(area_size) + [1, 1]

        # shape of croped image before resize
        temp_shape = array_shape - array_shape % steps + self.area_size + [1, 1]
        
        # final shape of croped resized image
        new_shape = temp_shape * np.array(self.search_grid) / np.array(area_size) + [1, 1]
        
        for (fx, fy) in itertools.product(range(steps[0]), range(steps[1])):
#            print('Processing image for ' + str((fx, fy)))
                
#            full_size = border_img.size
#            this_img = border_img.crop((fy, fx,
#                                        full_size[0] + fy - steps[0],
#                                        full_size[1] + fx - steps[1]))
            this_array = border_array[fx:fx + temp_shape[0], fy:fy + temp_shape[1]]
            out_img = Image.fromarray(this_array).resize(np.flipud(new_shape))
#            print('  this image size is ' + str(this_img.size))
            
            
            
#            border_img = ImageChops.offset(border_img, -fy, -fx)
#            border_img.show()
#            print('  resized border image size is : ' + str(this_size))
#            print('  scaling image by: ' + str(1. * np.flipud(self.search_grid) / np.flipud(area_size)))
            
            
#            out_img = border_img.resize(new_size, self.method)
#            out_img.show()
            self.arrays[fx, fy] = np.array(out_img)
#        pdb.set_trace()
    
    def _show_reshaped_images(self):
        for ar in self.arrays.values():
            pdb.set_trace()
            Image.fromarray(ar).resize(np.flipud(ar.shape[:2])).show()
    
    def extract_around(self, coord):
        area_size = self.area_size
        steps = self._steps(area_size, self.search_grid)
#        pdb.set_trace()
        new_coord, coord_image = self._coord_to_array_index_and_coord(coord, steps)
        image = self.arrays[tuple(coord_image)]
        border = self.search_grid / 2
        sub_ai = image[new_coord[0] - border[0]:new_coord[0] + border[0] + 1, new_coord[1] - border[1]:new_coord[1] + border[1] + 1]
        return sub_ai
    
    def _steps(self, area, res):
        '''
        :param area:
        :param res:
        '''
        return np.array(((area[0] - 1) / gcd(area[0] - 1, res[0] - 1), (area[1] - 1) / gcd(area[1] - 1, res[1] - 1)))
    
    def _coord_to_array_index_and_coord(self, coord, factor):
        new_coord = coord / factor * self.search_grid / 2 + self.search_grid / 2
#        new_coord = (coord - coord % factor) / (factor) + self.search_grid / 2
        coord_image = coord % factor
        return new_coord, coord_image
    
    def _add_border_pil(self, img, border_size):
        size = np.array(img.size)
        border_size = np.array(border_size)
        new_size = size + border_size * 2 + [1, 1]
        border_img = Image.new(img.mode, new_size)
        border_img.paste(img, tuple(border_size) + tuple(size + border_size))
        
        W = np.array([size[0], 0])
        H = np.array([0, size[1]])
        Bw = np.array([border_size[0], 0])
        Bh = np.array([0, border_size[1]])
        
        
        tlc = img.copy().crop(tuple(size - border_size) + tuple(size))
        border_img.paste(tlc, (0, 0))
        
        t = img.copy().crop(tuple(H - Bh) + tuple(size))
        border_img.paste(t, tuple(Bw) + tuple(border_size + W))
        
        trc = img.copy().crop(tuple(H - Bh) + tuple(H + Bw))
        border_img.paste(trc, tuple(W + Bw) + tuple(W + Bw + border_size))
        
        l = img.copy().crop(tuple(W - Bw) + tuple(size))
        border_img.paste(l, tuple(Bh) + tuple(border_size + H))
        
        r = img.copy().crop(tuple((0, 0)) + tuple(H + Bw))
        border_img.paste(r, tuple(border_size + W) + tuple(border_size + size + Bw))
        
        llc = img.copy().crop(tuple(W - Bw) + tuple(W + Bh))
        border_img.paste(llc, tuple(H + Bh) + tuple(border_size + H + Bh)) 
        
        b = img.copy().crop(tuple((0, 0)) + tuple(W + Bh))
        border_img.paste(b, tuple(border_size + H) + tuple(border_size + size + Bh))
        
        lrc = img.copy().crop(tuple((0, 0)) + tuple(border_size))
        border_img.paste(lrc, tuple(border_size + size) + tuple(border_size * 2 + size))
        
        return border_img
    
    def _add_border_array(self, array, border_size, offset=None):
        '''
        :param array: numpy array in valid format for Image.fromarray
        :param border_size: the (width, height) of the border
        '''
        img = Image.fromarray(array)
        border_img = self._add_border_pil(img, border_size)  
        if offset is not None:
            border_img = ImageChops.offset(border_img, offset[0], offset[1])   
        return np.array(border_img)

class FastInterpolatorDummy():
    def __init__(self):
        pass

if __name__ == '__main__':
    area_size = (297, 297)
    search_grid = (99, 99)
    intp = ImageInterpolatorFast()
    intp.set_resolution(search_grid, area_size)
    
    print intp._factor(area_size, search_grid)
    
    
    fd = urllib.urlopen("http://lambda.cds.caltech.edu/~adam/zoom1im1.png")
    image_file = cStringIO.StringIO(fd.read())
    im = Image.open(image_file)
    array = np.array(im)

    
    intp.reshape_image(array)
    i = 0
    for cy in range(40):
        for cx in range(40):
            sub_ai = intp.extract_around((cx, 80 + cy))
            Image.fromarray(sub_ai).resize((300, 300)).save('subim' + str(i) + '.png')
            i += 1
#    intp._show_reshaped_images()
#    intp._add_border_array(array, (2, 2))
    
    pdb.set_trace()
    factor = intp._factor((5, 5), (3, 3))
    for i in range(12):
        print intp._coord_to_array_index_and_coord((i, 0), factor)
    pdb.set_trace()
