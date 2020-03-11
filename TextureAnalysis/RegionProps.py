# -*- coding:utf-8 -*-

"""
Code modified on March 2020 by Maria Teresa Parreira @Instituto Superior Tecnico, Lisboa

"""

import numpy as np
import math
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage.measure import regionprops, shannon_entropy,label




class RegionProps:
    """
    Various region properties
    """

    def __init__(self, img, level_min=1, level_max=256, threshold=None):
        """
        initialize
        :param img: normalized image
        :param level_min: min intensity of normalized image
        :param level_max: max intensity of normalized image
        """

        
        self.img = img
        self.bin_img = (img>0)*1
        self.n_level = (level_max - level_min) + 1
        self.level_min = level_min
        self.level_max = level_max

        


        self.features = self._calc_features()

    def _calc_features(self):
        """
        calculate feature values
        :return: feature values
        """
        
        
        features = {}
        
        props = regionprops(self.bin_img,self.img)
        
        features['Area'] = props[0].area #Number of pixels of the region.
        
        features['BB Area'] = props[0].bbox_area #Number of pixels of bounding box.
        
        features['Centroid'] = props[0].centroid #Centroid coordinate tuple.
        
        features['Weighted Centroid'] = props[0].weighted_centroid 
        #Centroid coordinate tuple (row, col) weighted with intensity image.
        
        features['Centroid Divergence'] = np.linalg.norm(np.array(props[0].centroid) - np.array(props[0].weighted_centroid))
        
        features['Eccentricity'] = props[0].eccentricity 
        #Eccentricity of the ellipse that has the same second-moments as the region. The eccentricity is the ratio of the focal distance (distance between focal points).
        
        features['Equivalent Diameter'] = props[0].equivalent_diameter
        #The diameter of a circle with the same area as the region.
        
        features['Major Axis Length'] = props[0].major_axis_length 
        #The length of the major axis of the ellipse that has the same normalized second central moments as the region.
        
        tensor_minor = props[0].inertia_tensor_eigvals[-1]
        
        features['Minor Axis Length'] = 4 * math.sqrt(abs(tensor_minor))
        #The length of the minor axis of the ellipse that has the same normalized second central moments as the region.
        
        features['Max Intensity'] = props[0].max_intensity
        #Value with the greatest intensity in the region.
        
        features['Min Intensity'] = props[0].min_intensity
        #Value with the greatest intensity in the region.
        
        features['Mean Intensity'] = props[0].mean_intensity 
        #Value with the least intensity in the region.
        
        features['Orientation'] = props[0].orientation
        #Angle between the 0th axis (rows) and the major axis of the ellipse that has the same second moments as the region, ranging from -pi/2 to pi/2 counter-clockwise.
        
        features['Perimeter'] = props[0].perimeter
        #Perimeter of object which approximates the contour as a line through the centers of border pixels using a 4-connectivity.
        
        features['Solidity'] = props[0].solidity
        #Ratio of pixels in the region to pixels of the convex hull image.
        
        features['Entropy'] = shannon_entropy(self.img, base=2)
        #The Shannon entropy is defined as S = -sum(pk * log(pk)), where pk are frequency/probability of pixels of value k.
        
        features['Circularity'] = (4*props[0].area*math.pi)/(props[0].perimeter**2)
        #Circularity that specifies the roundness of objects.
        
        return features

    
    def print_features(self, print_values = True):
        """
        print features
        """
        
        if print_values:
            print("----RegionProps-----")
        feature_labels = []
        feature_values = []
        for key in self.features.keys():
            if print_values:
                print("{}: {}".format(key, self.features[key]))
            feature_labels.append(key)
            feature_values.append(self.features[key])
            
        return feature_labels, feature_values
