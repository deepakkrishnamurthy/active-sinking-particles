# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 23:18:43 2021
Generates plots of the rotational analysis data
@author: Deepak
"""
import pandas as pd
import seaborn as sns
import os

folder = 'C:/Users/Deepak/Dropbox/ActiveMassTransport_Vorticella_SinkingAggregates/RotationalAnalysis/FinalAnalysis'
analysis_file = 'RotationalAnalysis_combined.csv'

df = pd.read_csv(os.path.join(folder, analysis_file))

df['angular speed (1/s)'] = (df['angular velocity x']**2 + df['angular velocity y']**2 + df['angular velocity z']**2)**(1/2)
df['no:of vorticella'] = np.nan
print(df)
# No:of vorticella mapping
df_num_vorticella = pd.read_csv(os.path.join(folder, 'Sphere_Vorticella_numbers.csv'))


for ii in range(len(df_num_vorticella)):
    
    track_id = df_num_vorticella['track ID'][ii]
    
    df['no:of vorticella'].loc[df['track ID'] == track_id]