# -*- coding: utf-8 -*-
"""
Created on Tue May 28 13:56:14 2019

@author: hemax
"""

"kmeans to obtain the ground truth labels based on the total red intensity, total green intensity"
"and total blue/DAPI intensity" "didnt use this!!"

from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_pickle(r'')

#feature space for the k-means
matrix = df.as_matrix(columns = ['Total Red', 'Total Green', 'Total Intensity'])


X = matrix.astype(float).reshape(matrix.shape)

#obtain 2 clusters
kmeans = KMeans(n_clusters=2, n_init = 100, verbose = 0, tol = 1e-10, precompute_distances = True).fit(X)
  
aux = kmeans.labels_  # obtain the k-means predictions 


def threeD_plot(names, df, aux):


	import pandas as pd
	from pandas import DataFrame
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D

	#subset = df[df.Image.str.startswith(names)]

	subset = df

	color_lst = build_color_array(aux)

	threedee = plt.figure().gca(projection='3d')
	threedee.scatter(subset['Total Red'], subset['Total Green'], c = color_lst)
	threedee.set_xlabel('Total Red')
	threedee.set_ylabel('Total Green')
	threedee.set_zlabel('Total Blue')
	plt.show()

	return color_lst



def build_color_array(aux):
    color_list = []
    for r in aux:
		#color_list.append([np.around(r/255.0, decimals = 1), np.around(g/255.0, decimals = 1), 0.0])
        if(r==0):  
            color_list.append([1, 0, 0])
        elif(r==1):
            color_list.append([0, 1, 0])
        else:
            color_list.append([0, 0, 1])
       
    
    final_color_list = np.asarray(color_list)
    
    return final_color_list

threeD_plot(['All'], df, aux)



def normalized_red_green_plt_aux(names, df, total , aux):

	# (r, g) plot with normalized colors for each point
	
	color = ['k', 'g', 'y']
	fig, ax = plt.subplots()
	for i, name in enumerate(names):    
		
		subset = df[df.Image.str.startswith(name)]


		if total:
			x = subset["Total Red"].values
			y = subset["Total Green"].values

			red = subset["Normalized Total Red"].values
			green = subset["Normalized Total Green"].values


		else:    	
			x = subset["Mean Red"].values
			y = subset["Mean Green"].values


			red = subset["Normalized Mean Red"].values
			green = subset["Normalized Mean Green"].values
# =============================================================================
#         x_aux = subset["Mean Red"].values
#         y_aux = subset["Mean Green"].values
#         
#         x = x_aux / (x_aux + y_aux)
#         y = y_aux / (x_aux + y_aux)
# =============================================================================	
		color_lst = build_color_array(aux)
		
		#xi = np.linspace(np.min(x), np.max(x), 500)
		#yi = np.interp(xi, x, y, yp)
		#ax.plot(x, y, 'g'+'o', xi, yi, 'm'+'-')
		ax.scatter(x, y, c = color_lst)

		ax.set_title(name)
		ax.set_xlabel('Mean Red')
		ax.set_ylabel('Mean Green')



normalized_red_green_plt_aux(['All'], df, False, aux)


def area_intensity_plt_colored(names, df, aux):
	

	# (A, TI) plot with non-normalized colors for each point

	fig, ax = plt.subplots()
	for i, name in enumerate(names):    
		
		subset = df[df.Image.str.startswith(name)]
		
		
		x = subset["norm_area"].values
		y = subset["norm_intensity"].values

		
		color_lst = build_color_array(aux)
		
		#xi = np.linspace(np.min(x), np.max(x), 500)
		#yi = np.interp(xi, x, y, yp)
		#ax.plot(x, y, 'g'+'o', xi, yi, 'm'+'-')
		#ax.plot(x, y, c = color_lst)
		
		ax.scatter(x, y, c = color_lst)

		ax.set_title(name)
		ax.set_xlabel('Total Area')
		ax.set_ylabel('Total Intensity')



area_intensity_plt_colored(['All'], df, aux)


def area_intensity_plt_colored1(names, df, total):
	

	# (A, TI) plot with non-normalized colors for each point

	fig, ax = plt.subplots()
	for i, name in enumerate(names):    
		
		subset = df[df.Image.str.startswith(name)]
		
		
		x = subset["norm_area"].values
		y = subset["norm_intensity"].values
		

		if total:

			red = subset["Total Red"].values
			green = subset["Total Green"].values

		else:

			red = subset["Normalized Mean Red"].values
			green = subset["Normalized Mean Green"].values
		
		color_lst = build_color_array1(red * 255 , green * 255)
		
		#xi = np.linspace(np.min(x), np.max(x), 500)
		#yi = np.interp(xi, x, y, yp)
		#ax.plot(x, y, 'g'+'o', xi, yi, 'm'+'-')
		#ax.plot(x, y, c = color_lst)
		
		
		
		ax.scatter(x, y, c = color_lst)

		ax.set_title(name)
		ax.set_xlabel('Total Area')
		ax.set_ylabel('Total Intensity')



def build_color_array1(red, green):
	color_list = []
	for r, g in zip(red, green):
		#color_list.append([np.around(r/255.0, decimals = 1), np.around(g/255.0, decimals = 1), 0.0])
		color_list.append([r/255.0, g/255.0, 0.0])
	
	final_color_list = np.asarray(color_list)
	
	return final_color_list


area_intensity_plt_colored1(['All'], df, False)