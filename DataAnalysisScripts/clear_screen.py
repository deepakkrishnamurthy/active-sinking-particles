# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 10:57:30 2021

@author: Deepak
"""

import os, time

clear = lambda: os.system('cls')

for i in range(10,0,-1):
    
    clear()
    print(i)
    time.sleep(1)