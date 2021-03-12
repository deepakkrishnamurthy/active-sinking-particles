# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 10:49:25 2021

@author: Deepak
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
plt.close("all")

def velocity_central_diff(time, data):
        
    assert(len(time) == len(data))
    
    data_len = len(time)
    velocity = np.zeros(data_len)
    
    for i in range(data_len):
        if(i == 0):
            # Forward difference at the start points
            velocity[i] = (data[i+1] - data[i])/(time[i+1] - time[i])
        
        elif(i == data_len-1):
            # Backward difference at the end points
            velocity[i] = (data[i] - data[i-1])/(time[i] - time[i-1])
    
        else:
            # Central difference for all other points
            velocity[i] = (data[i+1] - data[i-1])/(time[i+1] - time[i-1])
            
    return velocity
                

pixelpermm = 449 # 4x objective
time_window = 10 # Time window over which we calculate the velocity in seconds

# Load the data file that contains the aggregate tracks
data_folder = 'C:/Users/Deepak/Dropbox/ActiveMassTransport_Vorticella_SinkingAggregates/TranslationAnalysis/AggregateCentroidTracks'
save_folder = 'C:/Users/Deepak/Dropbox/ActiveMassTransport_Vorticella_SinkingAggregates/TranslationAnalysis/AggregateCentroidTracks_velocity'
plots_folder = 'C:/Users/Deepak/Dropbox/ActiveMassTransport_Vorticella_SinkingAggregates/TranslationAnalysis/Plots'

files = os.listdir(data_folder)

for file in files:
    
    df = pd.read_csv(os.path.join(data_folder, file))
    
    # Subtract the image displacement contribution to the trajectory.
    X_stage = df['Xobj'] - df['Xobj_image']
    Z_stage = df['ZobjWheel'] - df['Zobj_image']
    
    # Add back the true centroid displacement of the object    
    X_obj_true = X_stage + df['sphere centroid X']/pixelpermm
    Z_obj_true = Z_stage + df['sphere centroid Z']/pixelpermm
    
    # Add this corrected data to the dataFrame and save it in a new location
    df['Xobj_true'] = X_obj_true
    df['Zobj_true'] = Z_obj_true
    
    # Take moving average of the centroid trajectory to remove pixel-noise due to tracking
    # Choose a window-size based on a time-interval 
    time_vect = np.array(df['Time'])
    delta_t_avg = np.nanmean(time_vect[1:] - time_vect[:-1])
    print('Mean time interval: {} s'.format(delta_t_avg))
    window_size = int(time_window/delta_t_avg)
    df['Xobj_true'] = np.array(df.loc[:,'Xobj_true'].rolling(window = window_size, center = True).mean())
    df['Zobj_true'] = np.array(df.loc[:,'Zobj_true'].rolling(window = window_size, center = True).mean())
    
    # Plot the final corrected trajectory
#    fig, (ax1) = plt.subplots(nrows = 1, ncols = 1,figsize = (8,8))
#    ax1.set_title(df['track ID'][0] + 'Corrected trajectory')
##    ax1.plot(df['Time'], df['Xobj'], 'r--', label = 'X displacement (raw)')
#    ax1.plot(df['Time'], df['Xobj_true'], 'r-', label = 'X displacement (corrected)')
#    ax1.legend()
#    ax1.set_xlabel('Time')
#    ax1.set_ylabel('X displacement (mm)')
#    plt.savefig(os.path.join(plots_folder, file[:-4] + '.png'), dpi = 300)
#    plt.show()
    
#    ax2.plot(df['Time'], df['ZobjWheel'], 'g--', label = 'Z displacement (raw)')
#    ax2.plot(df['Time'], df['Zobj_true'], 'g-', label = 'Z displacement (corrected)')
#    ax2.set_xlabel('Time')
#    ax2.set_ylabel('Z displacement (mm)')
#    ax2.legend()
#    plt.show()
    
    Xobj_true = df['Xobj_true']
    Zobj_true = df['Zobj_true']
    
    velocity_X = velocity_central_diff(time_vect, Xobj_true)
    velocity_Z = velocity_central_diff(time_vect, Zobj_true)
    
    df['Velocity X'] = velocity_X
    df['Velocity Z'] = velocity_Z
        
    velocity_X_fluctuation = velocity_X - np.nanmean(velocity_X)
    velocity_Z_fluctuation = velocity_Z - np.nanmean(velocity_Z)
    
    df['Velocity X fluctuation'] = velocity_X_fluctuation
    df['Velocity Z fluctuation'] = velocity_Z_fluctuation
    # Save the new data-frame
    df.to_csv(os.path.join(save_folder, file))