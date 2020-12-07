# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 12:04:41 2020
Multi-point tracking
@author: Deepak
"""

import cv2
import numpy as np
import imp
import GravityMachineTrack 
imp.reload(GravityMachineTrack)
import pandas as pd
import os
from scipy.ndimage.filters import uniform_filter1d

 # Contrast factor for image
clahe_cliplimit = 3.0

# Create a CLAHE object 
clahe = cv2.createCLAHE(clipLimit = clahe_cliplimit, tileGridSize = (6,6))

major_ver, minor_ver, subminor_ver = cv2.__version__.split('.')

tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

tracker_type = 'CSRT'
print('Using tracker: {}'.format(tracker_type))


# @@@ Implement support for daSiamRPN based trackers.

# Tracking parameters
n_points = 2    # No:of points tracked at a given time.

track_ids = [ii for ii in range(n_points)]

colors = {0:(255,0,0),1:(0,255,0),2:(0,0,255)}

Timestamp_array = []
track_id_array = []

bbox = {}
tracker_instance = {}
tracker_flag = {}
centroids = {}
centroids_x_array = {}
centroids_y_array = {}
image_names = []


def smooth_signal(data, window):      # Window is given in seconds
            
    return uniform_filter1d(data, size = window, mode="reflect")


def add_new_dict_entry(track_id):
    
    bbox[track_id] = []
    
    if tracker_type == 'BOOSTING':
        tracker_instance[track_id] = cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker_instance[track_id] = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker_instance[track_id] = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker_instance[track_id] = cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker_instance[track_id] = cv2.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        tracker_instance[track_id] = cv2.TrackerGOTURN_create()
    if tracker_type == 'MOSSE':
        tracker_instance[track_id] = cv2.TrackerMOSSE_create()
    if tracker_type == "CSRT":
        tracker_instance[track_id] = cv2.TrackerCSRT_create()
        
    tracker_flag[track_id] = 0
    centroids [track_id] = []
    
    centroids_x_array[track_id] = []
    centroids_y_array[track_id] = []
    
    

for ii in track_ids:    # Allocate dicts to store the tracks
    
    add_new_dict_entry(ii)
    
    


# Load the video or image folder and associated gravity machine track.

# No vorticella
#track_file = 'H:/2019 Monterey Trip/Vorticella_GM/2019_08_22_afterdinner/track6/track000.csv'
#sphere = 'Sphere011'
#T_start = 40
#T_end = 70
    
#track_file = 'H:/2019 Monterey Trip/Vorticella_GM/2019_08_22_afterdinner/track7/track000.csv'
#sphere = 'Sphere012'
#T_start = 180
#T_end = 210

# 3 vorticella
track_file = 'H:/2019 Monterey Trip/Vorticella_GM/2019_08_22/track9/track000.csv'
sphere = 'Sphere006'
T_start = 50
T_end = 80

# 7 vorticella
#track_file = 'H:/2019 Monterey Trip/Vorticella_GM/2019_08_22/track10/track000.csv'
#sphere = 'Sphere007'
#T_start = 65
#T_end = 95



track = GravityMachineTrack.gravMachineTrack(trackFile = track_file, Tmin = T_start, Tmax = T_end, findDims = True, flip_z = False, scaleFactor = 5)
save_file_name = sphere + 'RotationTracks_Tmin_{}_Tmax_{}'.format(T_start, T_end)

image_save_path = os.path.join('D:/Vorticella_GravityMachine/SphereRotation_analysis', save_file_name)

if(not os.path.exists(image_save_path)):
    os.makedirs(image_save_path)

stride = 10

image_index_subsampled = track.imageIndex[::stride]

TimeStamp_start = track.df['Time'][image_index_subsampled[0]]

nFrames = len(image_index_subsampled)

print("No:of frames in track: {}".format(nFrames))

overwrite = True
save = True
        
if (not os.path.exists(os.path.join(track.path, save_file_name)) or overwrite == True):
    # Display initial image and allow users to choose the n_points that are tracked
    imageindex = image_index_subsampled[0] 
                    
    image_name = track.df['Image name'][imageindex]
    
    image = cv2.imread(os.path.join(track.path, track.image_dict[image_name], image_name),0)
    image = clahe.apply(image)
    #cv2.imshow("Initial frame", image)
    #cv2.waitKey(0)
    
    
    # Choose the initial bounding boxes to start the tracking
    for ii in track_ids:
        bbox[ii] = cv2.selectROI(image, False)
    
        print('bbox {} chosen as {}'.format(ii, bbox[ii]))
        
         # Initialize tracker with first frame and bounding box
        tracker_flag[ii] = tracker_instance[ii].init(image, bbox[ii])
    
            
    
    # In a while loop try to track the n_points in subsequent frames
    counter = 0
    while True and counter < nFrames-1:
        
        imageindex = image_index_subsampled[counter] 
        image_name = track.df['Image name'][imageindex]
        TimeStamp = track.df['Time'][imageindex] - TimeStamp_start
        
        
        image = cv2.imread(os.path.join(track.path, track.image_dict[image_name], image_name),0)
        image = clahe.apply(image)
        
        image_raw = np.copy(image)
        # Start timer
        timer = cv2.getTickCount()
     
        # Update tracker
        for ii in track_ids:
            tracker_flag[ii], bbox[ii] = tracker_instance[ii].update(image)
            # Calculate the center of the bounding box and store it
            
            x_pos = bbox[ii][0] + bbox[ii][2]/2
            y_pos = bbox[ii][1] + bbox[ii][3]/2
            centroids[ii] = (x_pos, y_pos)
            centroids_x_array[ii].append(x_pos)
            centroids_y_array[ii].append(y_pos)
            
            
        
        Timestamp_array.append(TimeStamp)
       
        
        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
        
        for jj, ii in enumerate(track_ids):
            # Draw bounding box
            if tracker_flag[ii]:
                # Tracking success
                p1 = (int(bbox[ii][0]), int(bbox[ii][1]))
                p2 = (int(bbox[ii][0] + bbox[ii][2]), int(bbox[ii][1] + bbox[ii][3]))
                cv2.rectangle(image, p1, p2, (255,255,255), 2, 1)
                print('Tracking # {} succesfully'.format(ii))
            else :
                # Tracking failure
                cv2.putText(image, "Tracking failure detected for {}".format(ii), (100+20*jj,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(255,255,255),2)
                
                # Show the last detected bbox on the image
                p1 = (int(bbox[ii][0]), int(bbox[ii][1]))
                p2 = (int(bbox[ii][0] + bbox[ii][2]), int(bbox[ii][1] + bbox[ii][3]))
                cv2.rectangle(image_raw, p1, p2, (255,0,0), 2, 1)
                
                # Choose a new point to track
                new_id = max(track_ids)+1
                track_ids.append(new_id)
                add_new_dict_entry(new_id)
                
                bbox[new_id] = cv2.selectROI("Choose bbox for new track {}".format(new_id), image_raw, False)
                key = cv2.waitKey(0)
        
        # Display tracker type on frame
#        cv2.putText(image, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,50),2);
#             
#        # Display FPS on frame
#        cv2.putText(image, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,50), 2);
        
        # add timestamp
        
        cv2.putText(image, '{:.2f}'.format(np.round(TimeStamp, decimals = 2))+'s', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,50), 2);
         # Display result
        cv2.imshow("Tracking", image)
    #            cv2.imwrite(str(counter)+'.png',frame)
        # Exit if ESC pressed
        key = cv2.waitKey(1) & 0xff
        
        if(key == 27):
            # ESC. If the organism is detected correctly, then store the value and break from the loop
            break
        elif(key == 32):
            # SPACEBAR
            continue
        
        
        if(save):
            cv2.imwrite(os.path.join(image_save_path, '{:05d}'.format(counter) + '.tif'), image)
        counter += 1
        
    cv2.destroyAllWindows()
    # Store the point ID, position in memory/write it to file...
    
    # Store the track_ID (or point ID), position and time-stamps
    
    data = pd.DataFrame({'track ID':[], 'Time':[], 'centroid X':[], 'centroid Y':[]})
    
    for ii in track_ids:
        
        data = data.append(pd.DataFrame({'track ID':np.repeat(ii, len(centroids_x_array[ii]), axis = 0), 'Time': Timestamp_array, 'centroid X':centroids_x_array[ii], 'centroid Y':centroids_y_array[ii]}))
    
    if(save):
        data.to_csv(os.path.join(track.path, save_file_name+'.csv'))
    
    
    # Calculate the instantaneous velocity of the points
    velocity_x = {}
    velocity_y = {}
    speed = {}
    import matplotlib.pyplot as plt
    plt.figure()
    for ii in track_ids:
        x_centroids, y_centroids = np.array(centroids_x_array[ii]), np.array(centroids_y_array[ii])
        
        velocity_x[ii], velocity_y[ii] = np.array(x_centroids[1:]-x_centroids[:-1]), np.array(y_centroids[1:]-y_centroids[:-1])      
        speed[ii] = (velocity_x[ii]**2 + velocity_y[ii]**2)**(1/2)
        
        speed[ii] = smooth_signal(speed[ii], 5)
        
        plt.scatter(x_centroids[:-1], y_centroids[:-1], c = speed[ii])
        
    
    plt.show()
    
    
 
    
    
    
        
        
    
    
    
    # Overlay centrid locations and velocities on the image and save the overlaid images
    
#    for ii, index in enumerate(image_index_subsampled):
#        imageindex = image_index_subsampled[counter] 
#        image_name = track.df['Image name'][imageindex]
#        TimeStamp = track.df['Time'][imageindex]
#        
#        
#        image = cv2.imread(os.path.join(track.path, track.image_dict[image_name], image_name),0)
#        image = clahe.apply(image)
#        
#        x_centroids, y_centroids = centroids_x_array[ii], centroids_y_array[ii]
#    
#        for jj in track_ids:
            
            
        
        