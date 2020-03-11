# -*- coding:utf-8 -*-
"""
Code modified on March 2020 by Maria Teresa Parreira @Instituto Superior Tecnico, Lisboa


    GLHA
    Copyright (c) 2016 Tetsuya Shinaji
    This software is released under the MIT License.
    http://opensource.org/licenses/mit-license.php

    Date: 2016/01/29
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage


from scipy.stats import skew, kurtosis



class GLHA:
    """
    Gray Level Histogram Analysis
    """

    def __init__(self, img, level_min=1, level_max=256, threshold=None):
        """
        initialize
        :param img: normalized image
        :param level_min: min intensity of normalized image
        :param level_max: max intensity of normalized image
        """

        
        self.img = img
        self.n_level = (level_max - level_min) + 1
        self.level_min = level_min
        self.level_max = level_max

        hist,bin_edges = np.histogram(self.img.ravel(),256,[0,256])
        self.hist = np.array(hist)
        self.bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        self.features = self._calc_features()

    def _calc_features(self):
        """
        calculate feature values
        :return: feature values
        """

        features = {}
        
        
        features['Mean'] = np.mean(self.img)
        features['Std'] = np.std(self.img)
        features['Variance'] = np.var(self.img)
        features['Skewness'] = skew(self.img)
        features['Kurtosis'] = kurtosis(self.img)

        return features

    def print_features(self, print_values = True, show_figure=False):
        """
        print features
        :param show_figure: if True, show figure
        """
        
        if print_values:
            print("----GLHA-----")
        feature_labels = []
        feature_values = []
        for key in self.features.keys():
            if print_values:
                print("{}: {}".format(key, self.features[key]))
            feature_labels.append(key)
            feature_values.append(self.features[key])

        if show_figure:
            plt.plot(self.bin_centers, self.hist, 'o-b', label='Density')
            plt.plot([self.features['Mean'], self.features['Mean']],
                     [0, self.hist.max() * 1.2], '-r', label='Mean')
            plt.plot([self.features['Mean'] - self.features['Std'],
                      self.features['Mean'] - self.features['Std']],
                     [0, self.hist.max() * 1.2], '-.r', label='Std lower')
            plt.plot([self.features['Mean'] + self.features['Std'],
                      self.features['Mean'] + self.features['Std']],
                     [0, self.hist.max() * 1.2], '-.r', label='Std upper')
            plt.legend(loc=0, numpoints=1)
            plt.ylim(0, self.hist.max() * 1.2)
            plt.show()

        return feature_labels, feature_values


if __name__ == '__main__':
    pass