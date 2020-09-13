# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 16:28:46 2019

@author: Deepak
"""

import GravityMachineTrack
import imp
import msdanalyzer
import pandas as pd
import numpy as np
import os
imp.reload(GravityMachineTrack)
imp.reload(msdanalyzer)

import matplotlib.pyplot as plt
# %matplotlib inline

PixelPermm = {'4x':449, '10x':1122}
#------------------------------------------------------------------------------------------------------------------------------    
# Define some constants for the data analysis:

minTrackDuration = 30

    
    
# Folder in which to save analysis results
saveFolder = 'C:/Users/Deepak/Dropbox/ActiveMassTransport_Vorticella_SinkingAggregates/TrackAnalysis_results/MSD_analysis'
    
#analysis_file = 'C:/Users/deepak/Dropbox/GravityMachine/GravityMachineAnalysis_Scripts/Acantharia.csv'

#analysis_file = 'C:/Users/Deepak/Dropbox/GravityMachine/GravityMachineAnalysis_Scripts/TracksUsedForAnalysis/MSD_VelocityDistrib_TracksUsed/Acantharia.csv'

#analysis_file = 'C:/Users/Deepak/Dropbox/GravityMachine/GravityMachineAnalysis_Scripts/TracksUsedForAnalysis/MSD_VelocityDistrib_TracksUsed/CeratiumSp_noFork.csv'

#analysis_file = 'D:/Vorticella_GravityMachine/active-sinking-particles/TracksUsedForAnalysis/VorticellaGravityMachine/VelocityDistribution_BatchProcess.csv'

analysis_file = 'D:/Vorticella_GravityMachine/active-sinking-particles/TracksUsedForAnalysis/VorticellaGravityMachine/VelocityDistribution_BatchProcess_3.csv'

analysis_df = pd.read_csv(analysis_file)


#pairwise_conditions = [(x,y) for x in analysis_df['Organism'] for y in analysis_df['Condition']]

pairwise_conditions = [(analysis_df['Organism'][ii], analysis_df['Condition'][ii]) for ii in range(len(analysis_df))]

# Create unique pairs between the Organism and Condition columns
used = set()
unique_conditions = [x for x in pairwise_conditions if x not in used and (used.add(x) or True)]
print(50*'-')
print('Unique condition : {}'.format(unique_conditions))
print(50*'-')

ensemble_tracks = []
TrackArray = []

nUniqueConditions = len(unique_conditions)

overwrite = True

# Assemble the list of unique conditions i.e unique combinations of Organisms, Condition. Each Unique Condition will in general have several tracks

OrgDims = []

for ii in range(nUniqueConditions):
    
    curr_Organism = unique_conditions[ii][0]
    curr_Condition = unique_conditions[ii][1]
    
    bool_Org = analysis_df['Organism'] == curr_Organism
    bool_Cond = analysis_df['Condition'] == curr_Condition
    
 
    
    # Check if the analysis data is already present on disk for this Organism, Condition
    if(overwrite is True or not os.path.exists(os.path.join(saveFolder, curr_Organism+'_'+curr_Condition+'_MSD.csv')) or not os.path.exists(os.path.join(saveFolder, curr_Organism+'_'+curr_Condition))):
        
        tracks_sameCondition = analysis_df[bool_Org & bool_Cond]
        
        nTracksSameCondition = len(tracks_sameCondition)
        
        print('No of tracks with same condition: {}'.format(nTracksSameCondition))
        
        # Reset the track array: We only want to bin tracks from the same unique condition together
        TrackArray = []
        
        for jj in range(nTracksSameCondition):
            
            Track_df = tracks_sameCondition.iloc[jj]
            
            
            Objective = Track_df['Objective']
            
            
            assert(Objective in PixelPermm.keys())
            print('Pixel per mm: {}'.format(PixelPermm[Objective]))
            
            full_path = os.path.join(Track_df['rootFolder'],Track_df['trackFolder'], Track_df['trackFile'])
            
            print('Loading {}'.format(full_path))
            
            track = GravityMachineTrack.gravMachineTrack(trackFile = full_path , organism = Track_df['Organism'], condition = Track_df['Condition'], Tmin = Track_df['Tmin'], Tmax = Track_df['Tmax'], findDims = True, pixelPermm = PixelPermm[Objective], use_postprocessed = False)
            
                    
            # Filter tracks based on min Track Duration
            if(track.trackDuration >= minTrackDuration):
                TrackArray.append(track)

    
    
    
        msd1 = msdanalyzer.msdanalyzer(Tracks = TrackArray, ensemble_method = 'subtrack', Organism = curr_Organism, Condition = curr_Condition, savePath = saveFolder)
    
        msd1.computeSqDisp(save = True)
        msd1.computeMSD(save = True, overwrite = False)
    
        msd1.calculate_velocityDist()
        
    else:
        # If the precomputed trajectories and MSD data already exists then load them into memory
        
        msd1 = msdanalyzer.msdanalyzer(Tracks = None, ensemble_method = 'subtrack', Organism = curr_Organism, Condition = curr_Condition, savePath = saveFolder)
        
#        msd1.computeSqDisp(save = False, load = True)
        msd1.computeMSD(save = True, overwrite = False)

    # Non-linear least-squares fitting Including Correlated Error

    msd1.fitTrajectories(overwrite = True)

    # MSD plots

    msd1.plotMSD(figname = 1, plot_fit= True, savefig = True)

        
        
    
    
