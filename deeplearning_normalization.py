# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 14:37:52 2019

@author: hemaxi
"""

import pandas as pd


df = pd.read_pickle(r'D:\ISR\Teresa\nuclei_19_03.pickle')


def process_dataframe(df_aux):

	import numpy as np

	var = df_aux.groupby(['Image'], as_index = False).var()
	mean = df_aux.groupby(['Image'], as_index = False).mean()

	#area_std = var[var['Image'] == 'All_FS2_1_40xoil_60Zs_deconv_10_Mask.tif.png']['Area']

	normalized_total_intensity = []

	for index, row in df_aux.iterrows():
	    #print(row['Area'], row['Total Intensity'])
	    
	    
	    img_name = row['Image']
	    	    
	    
	    intensity_mean =  mean[mean['Image'] == img_name]['Mean Intensity']
	    
	    intensity_mean = intensity_mean.as_matrix()[0]
	    
	    intensity_var = var[var['Image'] == img_name]['Mean Intensity']norm
	    
	    intensity_var = intensity_var.as_matrix()[0]
	    
	    intensity = row['Mean Intensity']    
	    
	    n_intensity = (intensity - intensity_mean) / (np.sqrt(intensity_var) + 1e-5)
	    
	    normalized_total_intensity.append(n_intensity)
	    
# =============================================================================
# 	    print(img_name)
# 	    print(intensity_var)        
# 	    print(intensity_mean)        
# 	    print(n_intensity)
# =============================================================================

	import pandas as pd

	normalized_total_intensity_df = pd.Series((v for v in normalized_total_intensity))
	    
	    
	df_aux['norm_mean_intensity'] = normalized_total_intensity_df
	    
    
	return df_aux


df = process_dataframe(df)

a = df['Nucleus Patch'] * (df['norm_intensity']/df['Mean Intensity'])

df['Norm_Nucleus_Patch'] = a


df.to_pickle(r'D:\ISR\Teresa\nuclei_24_03.pickle')

