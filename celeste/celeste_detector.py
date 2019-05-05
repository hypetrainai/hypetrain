import matplotlib.pyplot as plt
import numpy as np
import imageio, scipy, glob, time

celeste_color_dict = {
    'red_hair' : np.array([179, 60, 69]).astype(np.float),
    'blue_hair' : np.array([74, 183, 255]).astype(np.float),
    'blue_hair_2' : np.array([99, 152, 218]).astype(np.float),
    'blue_hair_3' : np.array([81, 174, 252]).astype(np.float),
    'brown' : np.array([142, 64, 54]).astype(np.float),
    'blue_shirt' : np.array([99, 123, 255]).astype(np.float),
    'skin' : np.array([244, 215, 180]).astype(np.float)
}

class CelesteDetector():
    def __init__(self, color_dict, threshold = 10):
        self.colors = color_dict
        self.thres = threshold
        
    def detect(self, im, return_state = True):
        #state 0 is red hair
        #state 1 is blue hair
        #state 2 is other (probably white hair)
        #state -1 is no detection
        col = self.colors['blue_shirt']
        
        mask = self._get_mask(im, col)
        
        if np.sum(mask) == 0:
            if return_state:
                return (None, None, -1)
            return (None, None)
        
        y,x = self._COM_coordinates(mask)
        
        if return_state:
            
            
            state = self._get_state(im)
            return (y,x,state)
        return (y,x)
            
    def _get_state(self, im):
        if np.sum(self._get_mask(im, self.colors['red_hair']))>0:
            return 0
        if np.sum(self._get_mask(im, self.colors['blue_hair']))>0:
            return 1
        if np.sum(self._get_mask(im, self.colors['blue_hair_2']))>0:
            return 1
        if np.sum(self._get_mask(im, self.colors['blue_hair_3']))>0:
            return 1
        return 2
        

    def _get_mask(self, im, col):
        colarray = np.tile(np.expand_dims(np.expand_dims(col,0),0),[540,960,1]).astype(np.float)
        im_col = np.sqrt(np.sum((im - colarray)**2,2))
        mask = im_col<self.thres
        return mask
            
    def _COM_coordinates(self, mask):
        y,x = np.where(mask)
        return [np.mean(y),np.mean(x)]
        