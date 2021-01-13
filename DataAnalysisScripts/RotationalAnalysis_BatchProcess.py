# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 19:21:13 2021
Rotational analysis (Batch Process)
@author: Deepak
"""
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

pixelpermm = 449 # 4x objective
time_window = 3 # Time window over which we calculate the velocity in seconds

save_folder = 'C:/Users/Deepak/Dropbox/ActiveMassTransport_Vorticella_SinkingAggregates/RotationalAnalysis/FinalAnalysis/RotationalAnalysis_Fitting_Results'

# Load the feature and object centroid track files
tracks_folder = 'C:/Users/Deepak/Dropbox/ActiveMassTransport_Vorticella_SinkingAggregates/RotationalAnalysis/FinalAnalysis/FeatureTracks_Final'
track_files = os.listdir(tracks_folder)

for file in track_files:
    
    
    df = pd.read_csv(os.path.join(tracks_folder, file))
    
    print('Loaded file:{}'.format(file))
    
    R = int(df['object diameter (px)'][0]/2)
    
    print('radius of aggregate in pixels: {}'.format(R))
    
    nTracks = int(max(df['feature ID']+1))
    
    print('No:of features tracked: {}'.format(nTracks))
    
    nTimepoints = int(len(df['Time'])/nTracks)
    print(nTimepoints)
    
    # generate a dict containing time series for each track
    df_tracks = {ii : df.loc[df['feature ID']==ii] for ii in range(nTracks)}    
    time_vect = np.array(df_tracks[0]['Time']) # All tracks have a common time vector
    
    #--------------------------------------------------------------------------
    # Filter high-freq pixel-shift noise: Time windows for smoothing and velocity estimation
    #--------------------------------------------------------------------------
    # Choose a window-size based on a time-interval 
    
    delta_t_avg = np.nanmean(time_vect[1:]-time_vect[:-1])
    
    print('Mean time interval: {} s'.format(delta_t_avg))
    
    window_size = int(time_window/delta_t_avg)
    
    # make the window-size an even integer
    if window_size%2 !=0:
        window_size+=1
        
    print('Chosen window size for velocity calculation: {}'.format(window_size))
    
    sphere_centroid_X_filtered = None
    sphere_centroid_Z_filtered = None
    
    feature_centroid_X_filtered = {}
    feature_centroid_Z_filtered = {}
    
    for ii in range(nTracks):
        
        sphere_centroid_X_filtered = np.array(df_tracks[ii].loc[:,'sphere centroid X'].rolling(window = window_size, center = True).mean())
        sphere_centroid_Z_filtered = np.array(df_tracks[ii].loc[:,'sphere centroid Z'].rolling(window = window_size, center = True).mean())
        
        feature_centroid_X_filtered[ii] = np.array(df_tracks[ii].loc[:,'feature centroid X'].rolling(window = window_size, center = True).mean())
        feature_centroid_Z_filtered[ii] = np.array(df_tracks[ii].loc[:,'feature centroid Z'].rolling(window = window_size, center = True).mean())
    #     df_tracks[ii]['feature centroid X'] = df_tracks[ii]['feature centroid X'].rolling(window = window_size).mean()
    #     df_tracks[ii]['feature centroid Z'] = df_tracks[ii]['feature centroid Z'].rolling(window = window_size).mean()
    
    nan_mask = np.logical_not(np.isnan(sphere_centroid_X_filtered))
    
    image_centroids_x = {}
    image_centroids_z = {}
    centroids_x = {}
    centroids_z = {}
    
    # Choose a subset of the time-vect excluding the nan values
    time_vect = time_vect[nan_mask]
    nTimepoints = len(time_vect)
    
    # Sphere centroid common to all tracks
    sphere_centroids_x, sphere_centroids_z = sphere_centroid_X_filtered[nan_mask], sphere_centroid_Z_filtered[nan_mask]
    
    for ii in range(nTracks):
        image_centroids_x[ii], image_centroids_z[ii] = feature_centroid_X_filtered[ii][nan_mask], feature_centroid_Z_filtered[ii][nan_mask]
       
    #--------------------------------------------------------------------------
    # Get the centroid locations relative to the sphere center
    #--------------------------------------------------------------------------
    for ii in range(nTracks):
        centroids_x[ii] = image_centroids_x[ii] - sphere_centroids_x
        centroids_z[ii] = image_centroids_z[ii] - sphere_centroids_z
    #--------------------------------------------------------------------------
    # Calculate centroid velocity over a suitable time-window for each track
    #--------------------------------------------------------------------------

    overlap = int(window_size/2)  # Max overlap = window_size -1
    window_centers = np.array(range(int(window_size/2), nTimepoints-int(window_size/2), window_size - overlap))
    nWindows = len(window_centers) # No:of time-windows over the entire track
    
    window_edges = [[window_centers[ii] - int(window_size/2), window_centers[ii] + int(window_size/2)] for ii in range(nWindows)]
    print(window_centers)
    print(window_edges)
    
    centroids_velocity_x = {ii: np.zeros(nWindows) for ii in range(nTracks)}
    centroids_velocity_z = {ii: np.zeros(nWindows) for ii in range(nTracks)}
    positions_center_x = {ii:np.zeros(nWindows) for ii in range(nTracks)} # Position of the features at the mid-point of the time-windows
    positions_center_y = {ii:np.zeros(nWindows) for ii in range(nTracks)} # Position of the features at the mid-point of the time-windows
    positions_center_z = {ii:np.zeros(nWindows) for ii in range(nTracks)} # # Position of the features at the mid-point of the time-windows
    
    for index, center in enumerate(window_centers):
        
        # Create a sub-portion of the track containing N points
        for ii in range(nTracks):
            centroids_x_slice = centroids_x[ii][window_edges[index][0]:window_edges[index][1]]
            centroids_z_slice = centroids_z[ii][window_edges[index][0]:window_edges[index][1]]
            time_vect_slice = time_vect[window_edges[index][0]:window_edges[index][1]]
    
            poly_x = np.polyfit(time_vect_slice, centroids_x_slice, deg = 1)
            poly_z = np.polyfit(time_vect_slice, centroids_z_slice, deg = 1)
    
         
    
            centroids_velocity_x[ii][index] = poly_x[0]
            centroids_velocity_z[ii][index] = poly_z[0]
            
            positions_center_x[ii][index] = centroids_x[ii][center]
            positions_center_z[ii][index] = centroids_z[ii][center]
            positions_center_y[ii][index] = (R**2 - centroids_x[ii][center]**2 + centroids_z[ii][center]**2)**(1/2)
            
            # Calculate the tru translational velocities based on the known angular velocity
            r_init = np.array([positions_center_x[ii][index], positions_center_y[ii][index], positions_center_z[ii][index]])
    
    #--------------------------------------------------------------------------
    # Estimate the angular velocity at each time using least-squares fitting
    #--------------------------------------------------------------------------
    angular_velocity_fit = np.zeros((3, nWindows))
    angular_velocity_fit_mag = np.zeros(nWindows)
    
    for index in range(nWindows):
        
        # For each time instant build the matrix A and the data vector by looping over all the tracks
        matrix_A = np.zeros((2*nTracks, 3))
        data_vector_fit = np.zeros(2*nTracks)
        
        for ii in range(nTracks):
            r_x, r_y, r_z = positions_center_x[ii][index], positions_center_y[ii][index], positions_center_z[ii][index]
    
            matrix_A[2*ii,:] = [0, r_z, -r_y]
            matrix_A[2*ii +1, :] = [r_y, -r_x, 0]
    
            data_vector_fit[2*ii] = centroids_velocity_x[ii][index]
            data_vector_fit[2*ii+1] = centroids_velocity_z[ii][index]
    
    #     print(matrix_A)
    #     print(data_vector_fit)
        # Compute the least-squares solution using lin alg
    
        angular_velocity_fit[:, index] = np.matmul(np.linalg.pinv(matrix_A), data_vector_fit)
        
        angular_velocity_fit_mag[index] = np.sum(angular_velocity_fit[:, index]**2)**(1/2)
    #--------------------------------------------------------------------------
    # Save the data 
    #--------------------------------------------------------------------------
    time_vect_binned = time_vect[window_centers]
    R_mm = round(R/pixelpermm,3)
    df_analysis = pd.DataFrame({'Time (s)':time_vect_binned, 'track ID':np.repeat(df['track ID'][0],len(time_vect_binned), axis = 0), 
                                'track file':np.repeat(df['track file'][0],len(time_vect_binned), axis = 0), 'angular velocity x':angular_velocity_fit[0,:],
                                'angular velocity y':angular_velocity_fit[1,:],'angular velocity z':angular_velocity_fit[2,:], 
                                'object radius (mm)': np.repeat(R_mm,len(time_vect_binned), axis = 0)})
    
    df_analysis.to_csv(os.path.join(save_folder, "RotationalAnalysis_" + file))
    