#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 15:33:28 2019

@author: deepak
"""

from flowtrace import flowtrace
import os

#videoFolder = '/Volumes/My Book/2019 Monterey Trip/Tunicates/LarvaeReleaseImaging/2019_08_10/images'
#videoFolder = 'D:/Vorticella_GravityMachine/registered_images/'

#videoFolder = 'D:/Vorticella_GravityMachine/2019_08_22_Track5_CroppedImages'

#videoFolder = 'D:\Vorticella_GravityMachine\Flowtrace_analysis\RegisteredImages\Track5_640_670'

#videoFolder = 'D:/Vorticella_GravityMachine/PuertoRico_FieldData/vorticell_diatom/Registered/vorticell_diatom_0_113_gs'

#videoFolder = 'D:/Vorticella_GravityMachine/Flowtrace_analysis/2019_08_22_Track10'

#videoFolder = 'D:/Vorticella_GravityMachine/Flowtrace_analysis/2019_08_21_Afternoon_Track4'

videoFolder = 'D:/Vorticella_GravityMachine/Flowtrace_analysis/2019_08_22_AfterDinner_Track6'
 
saveFolder = os.path.join(videoFolder, 'FlowTrace')

if(not os.path.exists(saveFolder)):
    os.makedirs(saveFolder)

flowtrace(videoFolder,60,saveFolder, use_parallel=False, subtract_median=False)