# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 10:43:39 2019

@author: Hemaxi

Modified by Teresa @IST
"""

#import the necessary packages
import pandas as pd
import numpy as np
from TextureAnalysis.GLHA import GLHA
from TextureAnalysis.GLCM import GLCM
from TextureAnalysis.RegionProps import RegionProps
from matplotlib import pyplot as plt
from scipy import ndimage



#-----Read the pandas dataframe-----

#path = r'INPUT PATH'
path = r'C:\Users\Teresa\Desktop\TESE\Textural Analysis'

#df_name = r'NAME OF INPUT DF'
df_name = r'FINAL_normalized_PLUS_automatic_labels_02_06_19.pickle'
 
df_path = path +'\\' + df_name

df = pd.read_pickle(df_path)


#-----Calculate features-----


final_glha = []
final_glcm = []
final_props = []

for index, row in df.iterrows():

        patch = row['Nucleus Patch'] #obtain the nucleus patch
        patch_h = patch[patch>0] #gets only the values where intensity is !=0

        #calculate features
        glha = GLHA(patch_h, level_min=0, level_max=255, threshold=None)
        glcm = GLCM(patch)
        props = RegionProps(patch)
        
        #obtain feature labels (invariant) and their values
        glha_labels, glha_values = glha.print_features(print_values = False)
        glcm_labels, glcm_values = glcm.print_features(print_values = False)
        prop_labels, prop_values = props.print_features(print_values = False)
        
        
       #add each feature to its respective list, which will then be put into the dataframe
        for i in np.arange(len(glha_values)):
            if len(final_glha) < len(glha_values):
                    final_glha.append([glha_values[i]])
            else:
                final_glha[i].append(glha_values[i])
                
        for i in np.arange(len(glcm_values)):
            if len(final_glcm) < len(glcm_values):
                    final_glcm.append([glcm_values[i]])
            else:
                final_glcm[i].append(glcm_values[i])
                
        for i in np.arange(len(prop_values)):
            if len(final_props) < len(prop_values):
                    final_props.append([prop_values[i]])
            else:
                final_props[i].append(prop_values[i])
                
                
                

#print(final_glha)
#print(final_glcm)
#print(final_props)




#-----Add features to dataframe-----



for i in np.arange(len(glha_labels)):
    df[glha_labels[i]] = final_glha[i]
    
for i in np.arange(len(glcm_labels)):
    df[glcm_labels[i]] = final_glcm[i]

for i in np.arange(len(prop_labels)):
    df[prop_labels[i]] = final_props[i]


    
    
#-----Save the new dataframe-----

#path = r'OUTPUT PATH'
path = r'C:\Users\Teresa\Desktop\TESE\Textural Analysis'

#df_name = r'NAME OF NEW DF'
df_name = r'first_attempt.pickle'
 
df_path = path +'\\' + df_name

df.to_pickle(df_path)

#if wanting to save to xlsx file, uncomment the following:

#excel_name = r'first_attempt.xlsx'
#df_path =  path +'\\' + excel_name
#df.to_excel(df_path) 



