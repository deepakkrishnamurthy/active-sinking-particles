# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 12:36:25 2021
Draw tracks on images
@author: Deepak
"""
import pandas as pd
import seaborn as sns
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import cmocean

root_folder = 'C:/Users/Deepak/Dropbox/ActiveMassTransport_Vorticella_SinkingAggregates/RotationalAnalysis/FinalAnalysis/AnnotatedImages/Sphere009'

track_file = 'Sphere009_images_61_96.csv'

df = pd.read_csv(os.path.join(root_folder, track_file))

nTracks = int(max(df['feature ID']+1))
print('No:of features tracked: {}'.format(nTracks))
nTimepoints = int(len(df['Time'])/nTracks)

# generate a dict containing time series for each track
df_tracks = {ii : df.loc[df['feature ID']==ii] for ii in range(nTracks)}

time_vect = np.array(df_tracks[0]['Time']) # All tracks have a common time vector
time_vect = time_vect-time_vect[0]

image_centroids_x = {}
image_centroids_z = {}

centroids_velocity_x = {}
centroids_velocity_z = {}

for ii in range(nTracks):
    image_centroids_x[ii], image_centroids_z[ii] = np.array(df_tracks[ii]['feature centroid X']), np.array(df_tracks[ii]['feature centroid Z'])
    object_bbox_x, object_bbox_z = np.array(df_tracks[ii]['object bbox X']), np.array(df_tracks[ii]['object bbox Z'])
#    sphere_centroids_x, sphere_centroids_z = np.array(df_tracks[ii]['sphere centroid X']), np.array(df_tracks[ii]['sphere centroid Z'])
    
    df_tracks[ii]['centroids x'] = image_centroids_x[ii] - object_bbox_x
    df_tracks[ii]['centroids z']  = image_centroids_z[ii] - object_bbox_z
    

time_window = 3 # Time window over which we calculate the velocity in seconds

delta_t_avg = np.nanmean(time_vect[1:]-time_vect[:-1])

print('Mean time interval: {} s'.format(delta_t_avg))

window_size = int(time_window/delta_t_avg)

# make the window-size an even integer
if window_size%2 !=0:
    window_size+=1
    
print('Chosen window size for velocity calculation: {}'.format(window_size))



centroids_x_filtered = {}
centroids_z_filtered = {}

for ii in range(nTracks):
     
    centroids_x_filtered[ii] = np.array(df_tracks[ii].loc[:,'centroids x'].rolling(window = window_size, center = True).mean())
    centroids_z_filtered[ii] = np.array(df_tracks[ii].loc[:,'centroids z'].rolling(window = window_size, center = True).mean())

nan_mask = np.logical_not(np.isnan(centroids_x_filtered[0])) 

nan_mask = np.logical_and(nan_mask, time_vect<=30)

print(nan_mask)

# Choose a subset of the time-vect excluding the nan values
time_vect = time_vect[nan_mask]
nTimepoints = len(time_vect)

for ii in range(nTracks):
    centroids_x_filtered[ii] = centroids_x_filtered[ii][nan_mask]
    centroids_z_filtered[ii] = centroids_z_filtered[ii][nan_mask]
    

centroid_vel_x = {}
centroid_vel_z = {}
centroid_speed = {}
for ii in range(nTracks):

    centroid_vel_x[ii] = (centroids_x_filtered[ii][1:]-centroids_x_filtered[ii][:-1])/(time_vect[1:] - time_vect[:-1])
    centroid_vel_z[ii] = (centroids_z_filtered[ii][1:]-centroids_z_filtered[ii][:-1])/(time_vect[1:] - time_vect[:-1])
    
    centroid_speed[ii] = (centroid_vel_x[ii]**2 + centroid_vel_z[ii]**2)**(1/2)
    

images_path = os.path.join(root_folder, 'images')
comet_tail_length = 1000

plt.figure(figsize = (6,6), dpi=300)
for ii in range(nTimepoints-1):
    
    folder,image_file = os.path.split(df['image file'][ii])
    
    image = cv2.imread(os.path.join(images_path, image_file))
    
    plt.cla()
    plt.imshow(image, cmap = 'gray')
    
    start_index = max(0, ii - comet_tail_length)

    for track_id in range(nTracks):
        x_centroids, y_centroids = centroids_x_filtered[track_id][start_index:ii], centroids_z_filtered[track_id][start_index:ii]
        
        # Colored by speed
#        ax = plt.scatter(x_centroids, y_centroids, 50, c = centroid_speed[track_id][start_index:ii]/449, marker = 'o',cmap = cmocean.cm.amp)
        # Colored by time
        ax = plt.scatter(x_centroids, y_centroids, 20, c = time_vect[start_index:ii], marker = 'o',cmap = cmocean.cm.tempo)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
    
    plt.show()
    plt.pause(0.001)
    
fig = plt.gcf()
fig.colorbar(ax)

        