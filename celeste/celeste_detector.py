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
    def __init__(self, color_dict = celeste_color_dict, threshold = 10, search_delta_with_prior = 100):
        self.colors = color_dict
        self.thres = threshold
        self.sdwp = search_delta_with_prior
        self.death_clock_init = 10
        self.death_clock = self.death_clock_init
        if self.sdwp%2 == 1:
            self.sdwp += 1
        
    def detect(self, im, return_state = True, prior_coord = None):
        #state 0 is red hair
        #state 1 is blue hair
        #state 2 is other (probably white hair)
        #state 3 is death clock initiated
        #state -1 is no detection
        col = self.colors['blue_shirt']
        
        
        mask = self._get_mask(im, col, prior_coord = prior_coord)
        if np.sum(mask) == 0 and prior_coord is not None:
            mask = self._get_mask(im, col, prior_coord = None)
        if np.sum(mask) == 0:
                self.death_clock -= 1
                if return_state:
                    if self.death_clock > 0:
                        return (prior_coord[0], prior_coord[1], 3)
                    else:
                        return (None, None, -1)
                return (None, None)
        self.death_clock = self.death_clock_init    
        y,x = self._COM_coordinates(mask)
        if prior_coord is not None:
            miny = np.maximum(0,prior_coord[0]-self.sdwp//2)
            minx = np.maximum(0,prior_coord[1]-self.sdwp//2)
            y += miny
            x += minx
        if return_state:
            state = self._get_state(im, prior_coord)
            
            
            return (y,x,state)
        return (y,x)
            
    def _get_state(self, im, prior_coord):
        if np.sum(self._get_mask(im, self.colors['red_hair'], prior_coord))>0:
            return 0
        
        if np.sum(self._get_mask(im, self.colors['blue_hair'], prior_coord))>0:
            return 1
        if np.sum(self._get_mask(im, self.colors['blue_hair_2'], prior_coord))>0:
            return 1
        if np.sum(self._get_mask(im, self.colors['blue_hair_3'], prior_coord))>0:
            return 1
        return 2
        

    def _get_mask(self, im, col, prior_coord = None):
        
        if prior_coord is None:
            colarray = np.tile(np.expand_dims(np.expand_dims(col,0),0),[540,960,1]).astype(np.float)
            im_col = np.sqrt(np.sum((im - colarray)**2,2))
            mask = im_col<self.thres
        else:
            miny = np.maximum(0,prior_coord[0]-self.sdwp//2)
            minx = np.maximum(0,prior_coord[1]-self.sdwp//2)
            im_sub = im[miny:prior_coord[0]+self.sdwp//2, 
                                       minx:prior_coord[1]+self.sdwp//2]
            
            hs, ws, _ = im_sub.shape
            
            colarray = np.tile(np.expand_dims(np.expand_dims(col,0),0),[hs, ws,1]).astype(np.float)
            im_col = np.sqrt(np.sum((im_sub - colarray)**2,2))
            mask = im_col<self.thres
        return mask
            
    def _COM_coordinates(self, mask):
        y,x = np.where(mask)
        return [np.mean(y),np.mean(x)]
        