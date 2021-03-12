#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 15:33:28 2019

@author: deepak
"""

from flowtrace import flowtrace
import os

videoFolder = 'D:/Vorticella_GravityMachine/Flowtrace_analysis/FinalAnalysis/08-22-Track3_IMG_1113_IMG_1200'
saveFolder = os.path.join(videoFolder, 'FlowTrace/')
flowtrace(videoFolder, 30, saveFolder)