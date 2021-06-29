0#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 11:55:48 2019

Script to serially analyse gravity machine track files.

@author: deepak
"""
import imp
import GravityMachineTrack 
imp.reload(GravityMachineTrack)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, time

batch_file = 'D:/Vorticella_GravityMachine/active-sinking-particles/TracksUsedForAnalysis/VorticellaGravityMachine/VelocityDistribution_BatchProcess_4.csv'

df_batch = pd.read_csv(batch_file)

print(df_batch)

for ii in range(len(df_batch)):
    
    Tmin = df_batch['Tmin'][ii]
    Tmax = df_batch['Tmax'][ii]
    
    rootFolder = df_batch['rootFolder'][ii]
    trackFolder = df_batch['trackFolder'][ii]
    trackFile = df_batch['trackFile'][ii]
    
    FileName = os.path.join(rootFolder, trackFolder, trackFile)
    
    track_ID = df_batch['ID'][ii]
    Condition = df_batch['Condition'][ii]
    
   
    
    LocalTime = time.ctime(os.path.getmtime(FileName))
    
    PixelPermm = df_batch['Pixelpermm'][ii]
    
    track = GravityMachineTrack.gravMachineTrack(trackFile = FileName, organism = track_ID, condition = Condition, Tmin = Tmin, Tmax = Tmax, computeDisp = True, overwrite_piv = False, overwrite_velocity = False, findDims = True, localTime = LocalTime, flip_z = False, pixelPermm = PixelPermm, scaleFactor = 5)
    track.saveAnalysisData(overwrite = True)

#FileList = {}
#
#counter = 0
#while 1:
#
#    Tmin = float(input('Enter Tmin value: '))
#    Tmax = float(input('Enter Tmax value: '))
#    File = filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("CSV files","*.csv"),("all files","*.*")))
#    Organism = input('Enter organism name: ')
#    Condition = input('Enter track condition: ')
#    Size = float(input('Enter organism size in mm: '))
#    FileList[counter] = [Tmin, Tmax, File, Organism, Condition, Size]
#    counter += 1
#    
#    c = input('Press q now to exit, c to continue: ')
#    
#    print(c)
#
#    if c == 'q':
#        break
#
#    
#print(FileList)
#
#for ii in range(len(FileList)):
#    
#    Tmin = FileList[ii][0]
#    Tmax = FileList[ii][1]
#    FileName = FileList[ii][2]
#    Organism = FileList[ii][3]
#    Condition = FileList[ii][4]
#    Size = FileList[ii][5]
#    
#    print(Tmin)
#    print(Tmax)
#    print(FileName)
#    print(Organism)
#    print(Condition)
#    print(Size)
#    
#    track = GravityMachineTrack.gravMachineTrack(fileName = FileName, organism = Organism, condition = Condition, Tmin = Tmin, Tmax = Tmax, computeDisp = True, orgDim = Size, overwrite_piv=False, overwrite_velocity=False)
#
#    track.saveAnalysisData(overwrite = True)
#    
    
    





# Tmin = 0
# Tmax = 10

# File = '/Users/deepak/Dropbox/GravityMachine/DiatomTestDataset/track000.csv'
# ###



# # orgDim in mm
# track = GravityMachineTrack.gravMachineTrack(fileName = File, Tmin = Tmin, Tmax = Tmax, computeDisp = True, orgDim = 0.1, overwrite_piv=False, overwrite_velocity=True)



# track.saveAnalysisData(overwrite = True)




