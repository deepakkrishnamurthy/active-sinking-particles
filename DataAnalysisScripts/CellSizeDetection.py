#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 01:04:22 2019
Analysis of Pyro noctiluca expansion with time
@author: deepak
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cmocean
import os
import time
import scipy
import scipy.ndimage as ndimage
from roipoly import roipoly
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.lines import Line2D
import pickle
plt.close("all")
from PIL import Image
import imp
import Track
imp.reload(Track)
import numpy as np
import csv_tool as csv
import pandas as pd
import seaborn as sns
import matplotlib.colors as Colors
import rangeslider_functions
import cv2
import PIV_Functions
imp.reload(PIV_Functions)

def errorfill(x, y, yerr, color=None, alpha_fill=0.3, ax=None, label = None):
    ax = ax if ax is not None else plt.gca()
    if color is None:
        color = 'k'
#        color = ax._get_lines.color_cycle.next()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.plot(x, y, color=color, label = label)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)

class gravMachineTrack:

    def __init__(self, path, Tmin=0, Tmax=0):
        
        self.path = path
        
        self.openFile()
        
        self.imgFormat = '.svg'
        self.root, self.Organism = os.path.split(self.path)
        
        
        
        self.PIVfolder = os.path.join(self.path, 'PIVresults_64px')
        
        if(not os.path.exists(self.PIVfolder)):
            os.makedirs(self.PIVfolder)
        
        self.df = pd.read_csv(os.path.join(self.path, self.trackFile))
        
        self.ColumnNames = list(self.df.columns.values)
        
        print(self.ColumnNames)
        
        # X position relative to image center
        self.Xobj_name = self.ColumnNames[1]
        self.Yobj_name = self.ColumnNames[2]
        # Z position relative to image center
        self.Zobj_name = self.ColumnNames[3]
        
        self.df['Time'] = self.df['Time'] - self.df['Time'][0]
        
    
        # Crop the track based on the specified time limits
        if(Tmax==0):
            Tmax = np.max(self.df['Time'])
                    
        Tmin_index = next((i for i,x in enumerate(self.df['Time']) if x >= Tmin), None)
        Tmax_index = next((i for i,x in enumerate(self.df['Time']) if x >= Tmax), None)
          
        print(Tmin_index)
        print(Tmax_index)
                
        
        self.df = self.df[Tmin_index:Tmax_index]
        
        df_index = self.df.index.values
        
        
        df_index = df_index - df_index[0]
    
        self.df = self.df.set_index(df_index)
        
        
        self.df['ZobjWheel'] = self.df['ZobjWheel'] - self.df['ZobjWheel'][0]
        
        self.trackLen = len(self.df)
        
        # Find the average sampling frequency
        self.T = np.linspace(self.df['Time'][0],self.df['Time'][self.trackLen-1], self.trackLen)  # Create a equi-spaced (in time) vector for the data.

        #Sampling Interval
        self.dT = self.T[1]-self.T[0]
        self.samplingFreq = 1/float(self.dT)
        # Window to use for smoothing data. 
        # IMPORTANT: We only keep variations 10 times slower that the frame rate of data capture.
        self.window_time = 50*self.dT
        
        self.computeVelocity()
        self.computeAccln()
        
#        self.computeDisplacement()
        
        self.createImageIndex()
        
        self.setColorThresholds()
        
        
        # PIV parameters
        self.window_size = 64
        self.overlap = 32
        self.searchArea = 64
        
 
        self.pixelPermm =  314*(self.imW/720)   # Pixel per mm for TIS camera (DFK 37BUX273) and 720p images
        
        self.mmPerPixel = 1/self.pixelPermm
        
        print('Pixels per mm: {}'.format(self.pixelPermm))
        
        self.findOrgDims(circle=1)
        
        # Initialize a suitable tracker
        tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

        self.initializeTracker('CSRT')
        
        
    def openFile(self):
        print('Opening dataset ...')

        self.image_dict = {}
        
        trackFileNames = []
        if os.path.exists(self.path):

            # Walk through the folders and identify ones that contain images
            for dirs, subdirs, files in os.walk(self.path, topdown=False):
               
                root, subFolderName = os.path.split(dirs)
                    
                print(subFolderName[0:6])
                if('images' in subFolderName):
                   
                   for fileNames in files:
                       key = fileNames
                       value = subFolderName
                       self.image_dict[key]=value

                if(os.path.normpath(dirs) == os.path.normpath(self.path)):
                    for fileNames in files:
                        if('.csv' in fileNames):
                            trackFileNames.append(fileNames)

            if(len(trackFileNames)==0):
                raise FileNotFoundError('CSV track was not found!')      
            elif(len(trackFileNames)>1):
                print('More than one .csv file found!')
                
#                for ii, filename in enumerate(trackFileNames):
#                    
#                    print('{}: {} \n'.format(ii+1, filename))
#                    
#                print('Choose the file to use:')
#                file_no = int(input())
#                    
#                self.trackFile = trackFileNames[file_no-1]
                
                self.trackFile = 'track_mod.csv'
                print('Loaded {}'.format(self.trackFile))
                
#                
#                for tracks in trackFileNames:
#                    if('division' in tracks):
#                        self.trackFile = tracks
#                        print('Loaded {}'.format(self.trackFile))
#                    elif('mod' in tracks):
#                        self.trackFile = tracks
#                        print('Loaded {}'.format(self.trackFile))

            else:
                self.trackFile = trackFileNames[0]
                print('Loaded {}'.format(self.trackFile))
                
    def initializeTracker(self, tracker_type):
        major_ver, minor_ver, subminor_ver = cv2.__version__.split('.')
        
        print('Using tracker: {}'.format(tracker_type))
         
        if int(minor_ver) < 3:
            self.tracker = cv2.Tracker_create(tracker_type)
        else:
            if tracker_type == 'BOOSTING':
                self.tracker = cv2.TrackerBoosting_create()
            if tracker_type == 'MIL':
                self.tracker = cv2.TrackerMIL_create()
            if tracker_type == 'KCF':
                self.tracker = cv2.TrackerKCF_create()
            if tracker_type == 'TLD':
                self.tracker = cv2.TrackerTLD_create()
            if tracker_type == 'MEDIANFLOW':
                self.tracker = cv2.TrackerMedianFlow_create()
            if tracker_type == 'GOTURN':
                self.tracker = cv2.TrackerGOTURN_create()
            if tracker_type == 'MOSSE':
                self.tracker = cv2.TrackerMOSSE_create()
            if tracker_type == "CSRT":
                self.tracker = cv2.TrackerCSRT_create()
    
    def createImageIndex(self):
        # Create an index of all time points for which an image is available
        self.imageIndex = []
        for ii in range(self.trackLen):
            
            if(self.df['Image name'][ii] is not np.nan):
#                print(self.df['Image name'][ii])
                self.imageIndex.append(ii)
                
#        print(self.imageIndex)
                
        # Open the first image and save the image size
        imageName = self.df['Image name'][self.imageIndex[0]]
        
        image_a = cv2.imread(os.path.join(self.path,self.image_dict[imageName],imageName))
        
      
        self.imH, self.imW, *rest = np.shape(image_a)
                
            
    def findOrgDims(self, circle=0):
        # Finds the maximum dimensions of the organism 
        saveFile = 'orgDims.pkl'
        
        OrgMajDim = []
        OrgMinDim = []
        OrgDim = []
        overwrite = False
        if(not os.path.exists(os.path.join(self.path,saveFile)) or overwrite):
            
            
            fileList = self.df['Image name'][self.imageIndex[0:10]]
            # Calculate based on 100 images
            for file in fileList:
                
                image = cv2.imread(os.path.join(self.path,self.image_dict[file],file))
                
                
                orgContour = self.colorThreshold(image = image)
                
                if(orgContour is not None):
                
                    if(circle):
                        (x_center,y_center), Radius = cv2.minEnclosingCircle(orgContour)
                        center = (int(x_center), int(y_center))
                        plt.figure(1)
                        plt.clf()
                        cv2.circle(image,center, int(Radius),(0,255,0),2)
                        plt.imshow(image)
                        plt.pause(0.001)
                        plt.show(block=True)
                        
                        OrgMajDim.append(self.mmPerPixel*2*Radius)
                        OrgMinDim.append(self.mmPerPixel*2*Radius)
                        
                        OrgDim.append(self.mmPerPixel*(2*Radius))
                    else:
                        try:
                            ellipse = cv2.fitEllipse(orgContour)
                            OrgMajDim.append(self.mmPerPixel*ellipse[1][0])
                            OrgMinDim.append(self.mmPerPixel*ellipse[1][1])
                            OrgDim.append(self.mmPerPixel*(ellipse[1][1] + ellipse[1][0])/float(2))
                            
                            plt.figure(1)
                            plt.clf()
                            cv2.ellipse(image,box=ellipse,color=[0,1,0])
                            plt.imshow(image)
                            plt.pause(0.001)
                            plt.show(block=True)
                        except:
                            OrgMajDim.append(np.nan)
                            OrgMinDim.append(np.nan)
                            OrgDim.append(np.nan)
                            
                else:
                    continue
                            
            
            OrgDim_mean = np.nanmax(np.array(OrgDim))
            OrgMajDim_mean = np.nanmax(np.array(OrgMajDim))
            OrgMinDim_mean = np.nanmax(np.array(OrgMinDim))
            
            with open(os.path.join(self.path,saveFile), 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump((OrgDim_mean, OrgMajDim_mean, OrgMinDim_mean), f)
                
        else:
            with open(os.path.join(self.path,saveFile), 'rb') as f:
                OrgDim_mean, OrgMajDim_mean, OrgMinDim_mean = pickle.load(f)
        
        self.OrgDim = OrgDim_mean
        self.OrgMajDim = OrgMajDim_mean
        self.OrgMinDim = OrgMinDim_mean
        
        print('*'*50)
        print('Organism dimension {} mm'.format(self.OrgDim))
        print('Organism Major dimension {} mm'.format(self.OrgMajDim))
        print('Organism Minor dimension {} mm'.format(self.OrgMinDim))
        print('*'*50)

    def colorThreshold(self,image):
    
        thresh_low = np.array(self.threshLow,dtype='uint8')
        thresh_high = np.array(self.threshHigh,dtype='uint8')
        
   
        
        im_th = cv2.inRange(image, thresh_low, thresh_high)
        
        kernel3 = np.ones((3,3),np.uint8)
        kernel5 = np.ones((5,5),np.uint8)
        kernel7 = np.ones((7,7),np.uint8)
        kernel15 = np.ones((15,15),np.uint8)
        
        
        im_th = cv2.morphologyEx(im_th, cv2.MORPH_CLOSE, kernel5)
        
        im_th = cv2.morphologyEx(im_th, cv2.MORPH_CLOSE, kernel7)
        
        im_th = cv2.morphologyEx(im_th, cv2.MORPH_OPEN, kernel7)
        
        im_th = cv2.morphologyEx(im_th, cv2.MORPH_CLOSE, kernel15)
    
        
    #    plt.matshow (im_th , cmap=cm.Greys_r )
    #    plt.show()
        
        # Select the largest contour as the final mask
        cnts = cv2.findContours(im_th,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2]
        
        print('No:of contours:{}'.format(len(cnts)))
        
        if(len(cnts)>0):
            largestContour = max(cnts, key=cv2.contourArea)
        
            M = cv2.moments(largestContour)
            
            # Approximate the contour to 1% of its arcLength
            epsilon = 0.01*cv2.arcLength(largestContour,True)
            approxContour = cv2.approxPolyDP(largestContour,epsilon,True)
                    
            
    #        orgCentroid = np.array((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))
            
    #        ellipse = cv2.fitEllipse(approxContour)
                
                
                # in mm
        
            return approxContour
        
        else:
            return None
    
    def computeVelocity(self):
        self.Vx= np.zeros(self.trackLen)
        self.Vy= np.zeros(self.trackLen)
        self.Vz= np.zeros(self.trackLen)
        self.Vz_objLab = np.zeros(self.trackLen)
        self.Theta_dot = np.zeros(self.trackLen)
        
        for i in range(0,self.trackLen):
            
            if(i==0):
                # Forward difference at the start points
                self.Vx[i] = (self.df[self.Xobj_name][i+1]-self.df[self.Xobj_name][i])/(self.df['Time'][i+1]-self.df['Time'][i])
                self.Vy[i] = (self.df[self.Yobj_name][i+1]-self.df[self.Yobj_name][i])/(self.df['Time'][i+1]-self.df['Time'][i])
                self.Vz[i] = (self.df['ZobjWheel'][i+1]-self.df['ZobjWheel'][i])/(self.df['Time'][i+1]-self.df['Time'][i])
                
                self.Vz_objLab[i] = (self.df[self.Zobj_name][i+1]-self.df[self.Zobj_name][i])/(self.df['Time'][i+1]-self.df['Time'][i])

                self.Theta_dot[i] = (self.df['ThetaWheel'][i+1]-self.df['ThetaWheel'][i])/(self.df['Time'][i+1]-self.df['Time'][i])

            
            elif(i==self.trackLen-1):
                # Backward difference at the end points
                self.Vx[i] = (self.df[self.Xobj_name][i]-self.df[self.Xobj_name][i-1])/(self.df['Time'][i]-self.df['Time'][i-1])
                self.Vy[i] = (self.df[self.Yobj_name][i]-self.df[self.Yobj_name][i-1])/(self.df['Time'][i]-self.df['Time'][i-1])
                self.Vz[i] = (self.df['ZobjWheel'][i]-self.df['ZobjWheel'][i-1])/(self.df['Time'][i]-self.df['Time'][i-1])
                
                self.Vz_objLab[i] = (self.df[self.Zobj_name][i]-self.df[self.Zobj_name][i-1])/(self.df['Time'][i]-self.df['Time'][i-1])

                self.Theta_dot[i] = (self.df['ThetaWheel'][i]-self.df['ThetaWheel'][i-1])/(self.df['Time'][i]-self.df['Time'][i-1])

                
            else:
                # Central difference for all other points
                self.Vx[i] = (self.df[self.Xobj_name][i+1]-self.df[self.Xobj_name][i-1])/(self.df['Time'][i+1]-self.df['Time'][i-1])
                self.Vy[i] = (self.df[self.Yobj_name][i+1]-self.df[self.Yobj_name][i-1])/(self.df['Time'][i+1]-self.df['Time'][i-1])
                self.Vz[i] = (self.df['ZobjWheel'][i+1]-self.df['ZobjWheel'][i-1])/(self.df['Time'][i+1]-self.df['Time'][i-1])
                
                self.Vz_objLab[i] = (self.df[self.Zobj_name][i+1]-self.df[self.Zobj_name][i-1])/(self.df['Time'][i+1]-self.df['Time'][i-1])

                self.Theta_dot[i] = (self.df['ThetaWheel'][i+1]-self.df['ThetaWheel'][i-1])/(self.df['Time'][i+1]-self.df['Time'][i-1])

#            self.Vx.append((self.Xobj[i+1]-self.Xobj[i])/(self.Time[i+1]-self.Time[i]))
#            self.Vy.append((self.Yobj[i+1]-self.Yobj[i])/(self.Time[i+1]-self.Time[i]))
#            self.Vz.append((self.ZobjWheel[i+1]-self.ZobjWheel[i])/(self.Time[i+1]-self.Time[i])) 
            
    
        self.Vx= self.smoothSignal(self.Vx, window_time = self.window_time)
        self.Vy= self.smoothSignal(self.Vy, window_time = self.window_time)
        self.Vz= self.smoothSignal(self.Vz, window_time = self.window_time)
        
        self.Vz_objLab = self.smoothSignal(self.Vz_objLab, window_time = self.window_time)
        
        self.Theta_dot= self.smoothSignal(self.Theta_dot, window_time = self.window_time)

        
        
        
#        self.Vx, self.Vy, self.Vz = self.smoothVelocity(self.window_time)
        
        self.Speed = (self.Vx**2 + self.Vy**2 + self.Vz**2)**(1/2)
        
        self.Speed_z = (self.Vz**2)**(1/2)
        
    
    def computeAccln(self):
        
        self.Theta_ddot = np.zeros(self.trackLen)
        
        self.a_z = np.zeros(self.trackLen)
        
        for i in range(0,self.trackLen):
            
            if(i==0):
                # Forward difference at the start points
                
                self.a_z[i] = (self.Vz[i+1]-self.Vz[i])/(self.df['Time'][i+1]-self.df['Time'][i])
                self.Theta_ddot[i] = (self.Theta_dot[i+1]-self.Theta_dot[i])/(self.df['Time'][i+1]-self.df['Time'][i])

            
            elif(i==self.trackLen-1):
                # Backward difference at the end points
                self.a_z[i] = (self.Vz[i]-self.Vz[i-1])/(self.df['Time'][i]-self.df['Time'][i-1])
                self.Theta_ddot[i] = (self.Theta_dot[i]-self.Theta_dot[i-1])/(self.df['Time'][i]-self.df['Time'][i-1])
            else:
                # Central difference for all other points
                self.a_z[i] = (self.Vz[i+1]-self.Vz[i-1])/(self.df['Time'][i+1]-self.df['Time'][i-1])
                self.Theta_ddot[i] = (self.Theta_dot[i+1]-self.Theta_dot[i-1])/(self.df['Time'][i+1]-self.df['Time'][i-1])
    
        self.a_z = self.smoothSignal(self.a_z, self.window_time)
        self.Theta_ddot = self.smoothSignal(self.Theta_ddot, self.window_time)


        
        
        
    def computeDisplacement(self, x_data = None, y_data = None):
        # Compute the displacement by integrating a velocity time series.
        
        disp = scipy.integrate.cumtrapz(y = y_data, x = x_data)
        
        return disp
#        self.disp_z_computed = scipy.integrate.cumtrapz(y = self.Vz, x = self.df['Time'])
        
        

    def computeFluidVelocity(self, image_a, image_b, deltaT = 1, overwrite = False, masking = False, scaleFactor = 1.5):
        # Computes the mean fluid velocity, far away from objects, given a pair of images
                
        #--------------------------------------------------------------------------
        # Load the frame-pair into memory
        #--------------------------------------------------------------------------
        frame_a_color = cv2.imread(os.path.join(self.path, self.image_dict[image_a], image_a))
        frame_b_color = cv2.imread(os.path.join(self.path, self.image_dict[image_b], image_b))
        
        frame_a_gs = cv2.cvtColor(frame_a_color,cv2.COLOR_BGR2GRAY)

        #--------------------------------------------------------------------------
        # Perform the PIV computation
        #--------------------------------------------------------------------------
        saveFile = os.path.join(self.PIVfolder,'PIV_' + image_a[:-4]+'.pkl')
        
        
        if(not os.path.exists(saveFile) or overwrite):
            print('-'*50)
            print('Analyzing Frame pairs: {} and {} \n'.format(image_a,image_b))
            print('-'*50)
            x,y,u,v, sig2noise = PIV_Functions.doPIV(frame_a_color,frame_b_color, dT = deltaT, win_size = self.window_size, overlap = self.overlap, searchArea = self.searchArea)
            
            u, v = PIV_Functions.pivPostProcess(u,v,sig2noise, sig2noise_min=1.5, smoothing_param = 0)
            
            
            u,v = (PIV_Functions.data2RealUnits(data = u,scale = 1/(self.pixelPermm)), PIV_Functions.data2RealUnits(data = v,scale = 1/(self.pixelPermm)))
            
            #--------------------------------------------------------------------------
            # Threshold the image to extract the object regions
            #--------------------------------------------------------------------------
            if(masking):
                Contours = PIV_Functions.findContours(frame_a_color,self.threshLow,self.threshHigh,'largest')
            else:
                Contours = np.nan
            
            
            with open(saveFile, 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump((x, y , u, v, Contours), f)
            
        else:
            #--------------------------------------------------------------------------
            # Read the PIV data 
            #--------------------------------------------------------------------------
            print('-'*50)
            print('Loading: {} and {} \n'.format(image_a,image_b))
            print('-'*50)
            pklFile = saveFile
            x,y,u,v,Contours = PIV_Functions.readPIVdata(pklFile)
           
        
        
        
        
        
#        Centroids = PIV_Functions.findCentroids(Contours)
        
       
        
#        plt.figure()
#        plt.imshow(maskInsideCircle)
#        plt.show()
        
#        cv2.circle(frame_a_gs,(int(x_cent), int(y_cent)),int(scale_factor*radius), color = (0,255,0))
#        cv2.imshow('frame',frame_a_gs)
#        cv2.waitKey(1)
        
#        cv2.drawContours(frame_a_color, [Contours],0,(0,255,0),3)
#        cv2.imshow('frame',frame_a_color)
#        cv2.waitKey(1)

        
        
#        PIV_Functions.plotPIVdata(frame_a_gs,x,y,u,v, orgContour=Contours)
        
        
        if(masking is True):    
            x_cent, y_cent, radius = PIV_Functions.findCircularContour(Contours)
            maskInsideCircle = PIV_Functions.pointInCircle(x,y,x_cent,y_cent,scaleFactor*radius)
            u_farfield, v_farfield = (u[~maskInsideCircle], v[~maskInsideCircle])
        else:
            u_farfield, v_farfield = (u, v)
            
        # Find the mean velocity
        u_avg, v_avg = (np.nanmean(u_farfield), np.nanmean(v_farfield))
        u_std, v_std = (np.nanstd(u_farfield), np.nanstd(v_farfield))
        
        
        return u_avg, v_avg, u_std, v_std
 
      
    def FluidVelTimeSeries(self, overwrite = False):
        # Computes the Fluid velocity at each time point during which an image is available and stores the result.
        # For each pair of time-points with images:
        # Generate a mask for each image as necessary
        # Do PIV on the pair of images
        # Calculate the average velocity of the fluid in regions far from the object
        # Store the average velocity as a finction of time
        
        saveFile = 'fluidVelocityTimeSeries.pkl'
        
        savePath = os.path.join(self.path, saveFile)
        
        
        
        if(not os.path.exists(savePath) or overwrite):
        
            nImages = len(self.imageIndex)
#            nImages = 100
            
            n = min(nImages, len(self.imageIndex)-1)
            
            self.u_avg_array = np.zeros(n)
            self.v_avg_array = np.zeros(n)
            self.u_std_array = np.zeros(n)
            self.v_std_array = np.zeros(n)
            self.imageIndex_array = np.zeros(n, dtype='int')
            
            
            for ii in range(n):
                
               
                    
                imageindex_a = self.imageIndex[ii] 
                imageindex_b = self.imageIndex[ii + 1]
                
                image_a = self.df['Image name'][imageindex_a]
                image_b = self.df['Image name'][imageindex_b]
                
                
                # First check if both these images exist in memory
                try:
                    image_a_exists = os.path.exists(os.path.join(self.path, self.image_dict[image_a], image_a))
                except:
                    image_a_exists = False
                try:
                    image_b_exists = os.path.exists(os.path.join(self.path, self.image_dict[image_b], image_b))
                except:
                    image_b_exists = False
                                
                image_a_num = int(image_a[4:-4])
                image_b_num = int(image_b[4:-4])
                
                frame_gap = image_b_num - image_a_num
                
                print(frame_gap)
                
                
                
                if(image_a_exists and image_b_exists and frame_gap == 1):
                    print('Consequtive images found ...')
                
                    print(image_a)
                    print(image_b)
                    
                    dT = self.df['Time'][imageindex_b] - self.df['Time'][imageindex_a]
                    
                    self.u_avg_array[ii], self.v_avg_array[ii], self.u_std_array[ii], self.v_std_array[ii] = self.computeFluidVelocity(image_a,image_b,deltaT = dT, masking = True, scaleFactor=1.5, overwrite = False)
                    self.imageIndex_array[ii] = imageindex_a
                    
                # If either of those images do not exist, assume that the velocity remains constant over the missing frames
                elif(not image_a_exists or not image_b_exists):
                    print('One or more of image pair not found...')
                    print('Checking for next image index...')
                    self.u_avg_array[ii], self.v_avg_array[ii], self.u_std_array[ii], self.v_std_array[ii] = self.u_avg_array[ii-1], self.v_avg_array[ii-1], self.u_std_array[ii-1], self.v_std_array[ii-1] 
                    self.imageIndex_array[ii] = imageindex_a
                    continue
               
                
                
               
            
            
            self.u_avg_array, self.v_avg_array = (self.smoothSignal(self.u_avg_array, self.window_time),self.smoothSignal(self.v_avg_array, self.window_time))
            
            with open(savePath, 'wb') as f:  # Python 3: open(..., 'wb')
                    pickle.dump((self.imageIndex_array, self.u_avg_array, self.v_avg_array, self.u_std_array, self.v_std_array), f)
                
            
        else:
            with open(savePath, 'rb') as f:  # Python 3: open(..., 'wb')
                    self.imageIndex_array, self.u_avg_array, self.v_avg_array, self.u_std_array, self.v_std_array = pickle.load(f)
                
                
    def setColorThresholds(self):
        
        saveFile = 'colorThresholds.pkl'

        image_a = None
        
        imageName = self.df['Image name'][self.imageIndex[15677]]
        
        image_a = cv2.imread(os.path.join(self.path,self.image_dict[imageName],imageName))
        
      
        self.imH, self.imW, *rest = np.shape(image_a)
    
            
        
        print('Image Width: {} px \n Image Height: {} px'.format(self.imW, self.imH))
        
        if(not os.path.exists(os.path.join(self.path, saveFile))):
            
            print(os.path.join(self.path, self.image_dict[imageName],imageName))
            v1_min,v2_min,v3_min,v1_max,v2_max,v3_max = rangeslider_functions.getColorThreshold(os.path.join(self.path,self.image_dict[imageName],imageName))
            threshLow = (v1_min,v2_min,v3_min)
            threshHigh = (v1_max,v2_max,v3_max)
            
            with open(os.path.join(self.path,saveFile), 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump((threshLow, threshHigh), f)
        else:
            print('Color thresholds available! \n Loading file {} ...'.format(os.path.join(self.root,saveFile)))
            with open(os.path.join(self.path,saveFile), 'rb') as f:
                threshLow, threshHigh = pickle.load(f)
                
        self.threshLow = threshLow
        self.threshHigh = threshHigh
        
        print('Color thresholds for segmentation: \n LOW: {}, HIGH : {}'.format(self.threshLow, self.threshHigh))
    #--------------------------------------------------------------------------       
    # Signal Processing Functions
    #--------------------------------------------------------------------------       
    def smoothSignal(self, data, window_time):      # Window is given in seconds
            
            avgWindow = int(window_time*self.samplingFreq)
            
    #        if(pd.__version__ is not '0.20.3'):
    #            return data.rolling(avgWindow).mean()
    #        else:
            return pd.rolling_mean(data, avgWindow, min_periods = 1, center = True)
        
        
    def correctedDispVelocity(self, overwrite_flag = False):
        
        
        self.FluidVelTimeSeries(overwrite = overwrite_flag)
        
        # Vz is the object velocity relative to the stage V_objStage
        
        V_stageLab = -self.Vz[self.imageIndex_array] + self.Vz_objLab[self.imageIndex_array]
        
        
        
        V_fluidStage = self.v_avg_array - V_stageLab
        
        self.V_objFluid = self.Vz[self.imageIndex_array] - V_fluidStage
        
        self.corrected_disp =  self.computeDisplacement(x_data = self.df['Time'][self.imageIndex_array], y_data = self.V_objFluid)
        
    def detectCells(self, start_image=0, stop_image=0):
        
        
        organism_name = 'Pyro'
        
        root, track_name = os.path.split(self.path)
        
        trackData = pd.DataFrame({'Organism':[],'Track':[],'Cell number':[],'Time':[],'Image':[],'x_center':[],'y_center':[],'Area':[]})

        
#        saveFile = track_name+'_cell_division_data_'+start_image[:-4]+'_'+stop_image[:-4]+'.csv'
        saveFile = track_name + '_cell_division_data_start.csv'

        savePath = os.path.join(path, saveFile)
        
        overwrite = True
        
        if(not os.path.exists(savePath) or overwrite):
            
            start_index = np.flatnonzero(self.df['Image name'][self.imageIndex]==start_image)
        
            stop_index = np.flatnonzero(self.df['Image name'][self.imageIndex]==stop_image)
            
            print(start_index)
            print(stop_index)
            
            start_index = start_index[0]
            stop_index = stop_index[0]
            
            print(type(start_index))
            
            print(start_index)
            print(stop_index)
            
            # Open the first image so that the tracking box can be chosen
            
            imageindex_a = self.imageIndex[start_index] 
                
            image_a = self.df['Image name'][imageindex_a]
                
            img = cv2.imread(os.path.join(self.path, self.image_dict[image_a], image_a)) 
            # Uncomment the line below to select a different bounding box
            bbox = cv2.selectROI(img, False)
         
            # Initialize tracker with first frame and bounding box
            ok = self.tracker.init(img, bbox)
            
            imSize=200
    
            for ii in range(start_index, stop_index):
      
                imageindex_a = self.imageIndex[ii] 
                
                image_a = self.df['Image name'][imageindex_a]
                
                currTime = self.df['Time'][imageindex_a]
                
                print(image_a)
                
                img = cv2.imread(os.path.join(self.path, self.image_dict[image_a], image_a))
                
                ok, bbox = self.tracker.update(img)
                
                if(ok):
                    print('Tracking success!')
                    
                    x_pos = int(bbox[0] + bbox[2]/2)
                    y_pos = int(bbox[1] + bbox[3]/2)
                    
                    img_cropped = img[y_pos-int(imSize/2):y_pos+int(imSize/2),x_pos-int(imSize/2):x_pos+int(imSize/2)]
                    
                
                    # Convert to Grayscale
                    gray = cv2.cvtColor(img_cropped,cv2.COLOR_BGR2GRAY)
        
                    img_hsv = cv2.cvtColor(img_cropped,cv2.COLOR_BGR2HSV)
                    
                    
                    
                    
                    
                #    cv2.imshow('frame',gray)
                #    
                #    key = cv2.waitKey(0) & 0xFF
                #    # if the 'q' key is pressed, stop the loop
                #    if key == ord("q"):
                #        pass
        
                #    ret, thresh = cv2.threshold(gray,100,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                    
                  
                    
                    thresh = cv2.inRange(img_hsv, track.threshLow, track.threshHigh)
                    
                #    thresh = ~thresh
                #    cv2.imshow('frame',thresh)
                #    
                #    key = cv2.waitKey(0) & 0xFF
                #    # if the 'q' key is pressed, stop the loop
                #    if key == ord("q"):
                #        pass
                    
                    # noise removal
                    kernel = np.ones((3,3),np.uint8)
                    
                    
                    # Morphological closing
                    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,kernel, iterations = 4)
                    # Morphological opening
                    opening = cv2.morphologyEx(closing,cv2.MORPH_OPEN,kernel, iterations = 2)
                    
                    
                #    cnts = cv2.findContours(opening,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2]
                    
                #    for mm in range(len(cnts)):
                #        
                #        cv2.drawContours(opening, cnts, -1, (255,255,255), -1)
            
                #    cv2.imshow('Close + Open',opening)
                #    
                #    key = cv2.waitKey(0) & 0xFF
                #    # if the 'q' key is pressed, stop the loop
                #    if key == ord("q"):
                #        pass
        
                    # sure background area
                    
                    sure_bg = cv2.dilate(opening,kernel,iterations=3)
                    
                    
                #    cv2.imshow('Dilate',sure_bg)
                #    
                #    key = cv2.waitKey(0) & 0xFF
                #    # if the 'q' key is pressed, stop the loop
                #    if key == ord("q"):
                #        pass         
                    # Finding sure foreground area
                    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,3)
                    
                    
                    ret, sure_fg = cv2.threshold(dist_transform,0.85*dist_transform.max(),255,0)
                    
                    
                #    cv2.imshow('Distance transform',dist_transform)
            
                #    key = cv2.waitKey(0) & 0xFF
                #    # if the 'q' key is pressed, stop the loop
                #    if key == ord("q"):
                #        pass
                    
                    
#                    cv2.imshow('Frgnd',sure_fg)
#          
#                    key = cv2.waitKey(1) & 0xFF
#                    # if the 'q' key is pressed, stop the loop
#                    if key == ord("q"):
#                        pass
             
                #    # Finding unknown region
                    sure_fg = np.uint8(sure_fg)
                    unknown = cv2.subtract(sure_bg,sure_fg)
                #    
                #    cv2.imshow('bcknd',sure_bg)
                ##    
                #    cv2.imshow('fgnd',sure_fg)
                ##    
                #    cv2.imshow('unkwn',unknown)
                #    
                #    key = cv2.waitKey(0) & 0xFF
                #
                #    if key == ord("q"):
                #        pass
                    
                    
                    # Marker labelling
                    ret, markers = cv2.connectedComponents(sure_fg)
                    
                    # Add one to all labels so that sure background is not 0, but 1
                    markers = markers+1
                    
                    # Now, mark the region of unknown with zero
                    markers[unknown==255] = 0
                    
                    
                #    plt.figure(1)
                #    plt.cla()
                #    plt.imshow(markers,cmap=plt.get_cmap('inferno'))
                #    plt.show(block=False)
                    
                    markers = cv2.watershed(img_cropped,markers)
                    img_cropped[markers == -1] = [255,0,0]
                    
                    plt.figure(1)
                    plt.clf()
                    ax = plt.imshow(markers,cmap=plt.get_cmap('inferno'))
                    plt.colorbar(ax)
                    plt.title('After segmentation')
                    plt.pause(0.001)
                    plt.show(block=False)
                #    
                #    
                #    print(np.max(markers))
                    
                    # Total number of objects minus background
                    nObjs = np.max(markers)-1
                    
                    
                #    print(nObjs)
                    
                    nObjs = 2
                    
                    color = [(255,255,0),(0,255,255)]
                    
                
                    
                #    plt.imshow(markers, plt.get_cmap('inferno'))
                #    plt.pause(0.001)
                #    plt.show()
                    Color = ['r','b']
                #    plt.figure(2)
                #    plt.cla()
                    for jj in range(nObjs):
                        
                        cell = np.array(markers == jj+2, dtype='uint8')
                        
                        cnts = cv2.findContours(cell,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2]
                        print('No:of contours:{}'.format(len(cnts)))
                #
                        if(len(cnts)>0):
                            largestContour = max(cnts, key=cv2.contourArea)
                            ((x_center, y_center), radius) = cv2.minEnclosingCircle(largestContour)
                            area = np.pi*radius**2
    #                        M = cv2.moments(largestContour)
    #                        if M["m00"] != 0:
    #                
    #                            x_center = int(M['m10']/M['m00'])
    #                            y_center = int(M['m01']/M['m00'])
    #                            
    #                        else:
    #                            x_center, y_center = (np.nan, np.nan)
    #                        
    #                        area = cv2.contourArea(largestContour)
                            
                            cv2.drawContours(img_cropped, [largestContour],0,color[jj],1)
                            
                        else:
                            x_center, y_center = (np.nan, np.nan)
                            area = np.nan
                            
                #        plt.scatter(x_center, y_center, 10, color = Color[jj])
                        
                        trackData = trackData.append(pd.DataFrame({'Organism':[organism_name],'Track':[track_name],'Cell number':[jj+1],'Time':[currTime],'Image':[image_a],'x_center':[x_center],'y_center':[y_center],'Area':[area]}))
                        
                
    #                cv2.imshow('frame',img_cropped)
    #                key = cv2.waitKey(0) & 0xFF
    #            #
    #                if key == ord("q"):
    #                    pass
    #                
                #    plt.imshow(img_clahe, cmap = plt.get_cmap('viridis'))
                #    plt.pause(0.0001)
                #    plt.show()
                    
                        
                #    plt.figure(2)
                #    plt.cla()
                #    plt.imshow(img)
                #    plt.scatter(x_center,y_center,20)
                #    plt.pause(0.001)
                #    plt.show()
                
            else:
                print('Tracking failure detected!')
            
            trackData.to_csv(savePath)
            
            
            
        else:
            # Data already exists
            print('Data exists, Loading it now...')
            trackData = pd.read_csv(savePath)
            
        return trackData
           
    def detectCells_tracking(self, start_image, stop_image):
        
        
                
                
#        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12,12))
                
                
        
        
        
       
        
        imSize = 75
        
#        imW = 
#        imH = 280
        
        organism_name = 'Pyro'
        
        root, track_name = os.path.split(self.path)
        
        trackData = pd.DataFrame({'Organism':[],'Track':[],'Cell number':[],'Time':[],'Image':[],'x_center':[],'y_center':[],'Area':[]})

        

        
#        saveFile = track_name+'_cell_division_data_'+start_image[:-4]+'_'+stop_image[:-4]+'.csv'
        saveFile = track_name+'_cell_division_data_middle.csv'

        savePath = os.path.join(path, saveFile)
        
        overwrite = True
        
        if(not os.path.exists(savePath) or overwrite):
            
            start_index = np.flatnonzero(self.df['Image name'][self.imageIndex]==start_image)
        
            stop_index = np.flatnonzero(self.df['Image name'][self.imageIndex]==stop_image)
            
            print(start_index)
            print(stop_index)
            
            start_index = start_index[0]
            stop_index = stop_index[0]
            
            print(type(start_index))
            
            print(start_index)
            print(stop_index)
            
            
            # Open the first image so that the tracking box can be chosen
            
            imageindex_a = self.imageIndex[start_index] 
                
            image_a = self.df['Image name'][imageindex_a]
                
            img = cv2.imread(os.path.join(self.path, self.image_dict[image_a], image_a)) 
            # Uncomment the line below to select a different bounding box
            bbox = cv2.selectROI(img, False)
         
            # Initialize tracker with first frame and bounding box
            ok = self.tracker.init(img, bbox)
                
            step = 1
  
            for ii in range(start_index, stop_index, step):
                
                imageindex_a = self.imageIndex[ii] 
                
                currTime = self.df['Time'][imageindex_a]
                
                print(currTime)
                
                image_a = self.df['Image name'][imageindex_a]
                
                img = cv2.imread(os.path.join(self.path, self.image_dict[image_a], image_a))
                
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
            
                
                gray = cv2.medianBlur(gray,5)
                
#                gray = clahe.apply(gray)
              
    
                ok, bbox = self.tracker.update(img)
                
                if(ok):
                    print('Tracking success!')
                    
                    x_pos = int(bbox[0] + bbox[2]/2)
                    y_pos = int(bbox[1] + bbox[3]/2)
                    
                    img_cropped = gray[y_pos-int(imSize/2):y_pos+int(imSize/2),x_pos-int(imSize/2):x_pos+int(imSize/2)]
                    
#                    img_cropped = gray[y_pos-int(imH/8):y_pos+int(7*imH/8),x_pos-int(imW/2):x_pos+int(imW/2)]

#                    plt.figure(1)
#                    plt.clf()
#                    plt.imshow(img_cropped,cmap = cmocean.cm.gray)
#                    plt.pause(0.001)
#                    plt.show(block=False)
#                    
#                    cv2.imwrite(image_a,img_cropped)
                    
                    
                    
                    
                    ret, thresh = cv2.threshold(img_cropped,100,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                    
                    kernel = np.ones((3,3),np.uint8)
                
                
                    # Morphological closing
                    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,kernel, iterations = 5 )
                    # Morphological opening
                    opening = cv2.morphologyEx(closing,cv2.MORPH_OPEN,kernel, iterations = 2)
                    
                    
                    
                    cnts =  cv2.findContours(opening,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2]
                    
                    if(len(cnts)>0):
                        Contour = max(cnts, key=cv2.contourArea)
                        
                        ((x_center, y_center), radius) = cv2.minEnclosingCircle(Contour)
                        area = np.pi*radius**2
                        
                        cv2.circle(img_cropped,(int(x_center), int(y_center)),int(radius), (255,255,255),2)
                        cv2.imshow('Detected circle', img_cropped)
    #                
                        key = cv2.waitKey(1) & 0xFF
                    
                        if key == ord("q"):
                            break
#                        Time_series.append(currTime)
#                        Area_series.append(np.pi*radius**2)
    
                    
              
                    
                    trackData = trackData.append(pd.DataFrame({'Organism':[organism_name],'Track':[track_name],'Cell number':[1],'Time':[currTime],'Image':[image_a],'x_center':[x_center],'y_center':[y_center],'Area':[area]}))
                    
                   
                
                
                else:
                    print('Tracking failure detected!')
                    
                    
            trackData.to_csv(savePath)
                
        else:
            
            trackData = pd.read_csv(savePath)
            
            
        
        
        return trackData

            

        
# Pyro division tracks
cv2.destroyAllWindows()
#        
#path = '/Volumes/DEEPAK-SSD/Pyro_Division_Tracks/1121'
##
#Tmin = 0
#Tmax = 0
####
#track = gravMachineTrack(path, Tmin, Tmax)





###
#start_image = 'IMG_0018650.tif'
#stop_image  = 'IMG_0018895.tif'
#
##start_image = 'IMG_0019600.tif'
##stop_image  = 'IMG_0019999.tif'
#
#start_image = 'IMG_0020000.tif'
#stop_image  = 'IMG_0020119.tif'

#start_image = 'IMG_0020119.tif'
#stop_image  = 'IMG_0020164.tif'

#start_image = 'IMG_0018650.tif'
#stop_image  = 'IMG_0019499.tif'

#start_image = 'IMG_0020173.tif'
#stop_image  = 'IMG_0021571.tif'

#trackData = track.detectCells_tracking(start_image, stop_image)

#trackData = track.detectCells(start_image, stop_image)

#sns.lineplot(x='Time',y='Area',hue='Cell number',data = trackData)
#plt.xlabel('Time (s)')
#plt.ylabel('Area(px^2)')

#Time_series = trackData['Time']
#Area = trackData['Area']

#==============================================================================
# Plots of cell size vs time using the assimilated data
#==============================================================================
path1 = '/Volumes/DEEPAK-SSD/Pyro_Division_Tracks/1121'  # cell 2 tracked over long time

path2 = '/Volumes/DEEPAK-SSD/Pyro_Division_Tracks/1130div' # cell 1 tracked over long times

file1 = '1121_cell_division_data_Full_v3.csv'
file2 = '1130div_cell_division_data_Full_v3.csv'
trackData1 = pd.read_csv(os.path.join(path1, file1))

trackData2 = pd.read_csv(os.path.join(path2, file2))



# 1121
Time_2 = trackData1.loc[trackData1['Cell number']==2,'Time'] 
Area_2 = trackData1.loc[trackData1['Cell number']==2,'Area']
Area_21 = trackData1.loc[trackData1['Cell number']==1,'Area']
Time_21 = trackData1.loc[trackData1['Cell number']==1,'Time']

# 1130
Time_1 = trackData2.loc[trackData2['Cell number']==1,'Time']
Area_1 = trackData2.loc[trackData2['Cell number']==1,'Area']
Area_12 = trackData2.loc[trackData2['Cell number']==2,'Area']
Time_12 = trackData2.loc[trackData2['Cell number']==2,'Time']


T_split2 = 6750+1357 # 1121
T_split1 = 6750 #1130
Time_2 = Time_2 - T_split2
Time_21 = Time_21 - T_split2
Time_1 = Time_1 - T_split1
Time_12 = Time_12 - T_split1
#
#Area_dim = Area*(1/314)**2

#Radius = (Area_dim/np.pi)**(1/2)        # Radius in mm

cmap = cmocean.cm.matter

color = cmap(np.linspace(0,1,4))

# Plot of area in pixels
plt.figure()

#plt.plot(Time_12, Area_12, color = color[0],label = 'Track 1, Cell 2', alpha = 0.8,linewidth=3)
plt.plot(Time_1, Area_1, color = color[1],label = 'Track 1, Cell 1',alpha = 0.8,linewidth=3)
plt.plot(Time_21, Area_21, color = color[2],label = 'Track 2, Cell 1',alpha = 0.8,linewidth=3)
plt.plot(Time_2, Area_2, color = color[3],label = 'Track 2, Cell 2',alpha = 0.8,linewidth=3)
plt.xlabel('Time (s)')
plt.ylabel('Area (px^2)')
plt.legend()
#plt.xlim(230,830)
plt.ylim(500,1750)

plt.show()


# Plot of Cell radius in um

Area2Radius = 1000*(1/np.pi)**(1/2)*(1/314)

Radius_1 = (Area_1)**(1/2)*Area2Radius
Radius_2 = (Area_2)**(1/2)*Area2Radius
Radius_21 = (Area_21)**(1/2)*Area2Radius
Radius_12 = (Area_12)**(1/2)*Area2Radius

Radius_1 = pd.rolling_mean(Radius_1,30,center=True)
Radius_2 = pd.rolling_mean(Radius_2,30,center=True)
Radius_12 = pd.rolling_mean(Radius_12,30,center=True)
Radius_21 = pd.rolling_mean(Radius_21,30,center=True)



plt.figure()

plt.plot(Time_12, 2*Radius_12 , color = color[0],label = 'Track 1, Cell 2', alpha = 0.8,linewidth=3)
plt.plot(Time_1, 2*Radius_1, color = color[1],label = 'Track 1, Cell 1',alpha = 0.8,linewidth=3)
plt.plot(Time_21, 2*Radius_21, color = color[2],label = 'Track 2, Cell 1',alpha = 0.8,linewidth=3)
plt.plot(Time_2, 2*Radius_2, color = color[3],label = 'Track 2, Cell 2',alpha = 0.8,linewidth=3)
plt.xlabel('Time (s)')
plt.ylabel('Cell diameter (um)')
plt.legend()
#plt.xlim(0,1000)
#plt.ylim(75,150)

plt.show()


#==============================================================================



