# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 11:50:09 2020

Create plots involving ensembles over several distinct datasets.

@author: Deepak
"""

import pandas as pd
import numpy as np
import os
import seaborn as sns
import cmocean
from matplotlib import rcParams
from matplotlib import rc
import matplotlib.pyplot as plt

rc('font', family='sans-serif') 
rc('font', serif='Helvetica') 
rc('text', usetex='false') 
rcParams.update({'font.size': 24})


figures_folder = 'C:/Users/Deepak/Dropbox/ActiveMassTransport_Vorticella_SinkingAggregates/TrackAnalysis_results/Plots'

if(not os.path.exists(figures_folder)):
    os.makedirs(figures_folder)

folder = 'C:/Users/Deepak/Dropbox/ActiveMassTransport_Vorticella_SinkingAggregates/TrackAnalysis_results/MSD_analysis/TaylorFunctionFitting_Analysis11-Sep-2020'

files = os.listdir(folder)

data = pd.DataFrame()

for file in files:
    
    data = data.append(pd.read_csv(os.path.join(folder, file)), ignore_index = True)

print(len(data))
# Set the colors
#cmap = plt.get_cmap("tab20")
cmap = cmocean.cm.algae
cmap_new = []
ColorStyle={}
for ii in np.linspace(int(0),int(cmap.N),max(data['Condition'])+1,dtype='int'):
    cmap_new.append(cmap(ii))

ColorStyle = cmap_new

print(len(ColorStyle))

print(ColorStyle)
#for ii, org in enumerate(Organisms):
#    ColorStyle[org] = cmap_new[ii]
    
# Set the markers
MarkerStyle = {0:'o',1:'s', 2:'.', 3:'d',4:'^',5:'p',6:'v',7:'*'}


#==============================================================================
# ==============================  PLOTS  ======================================
#==============================================================================
# Fit a Stokes law curve to the data to estimate the mean density mismatch for all aggregates

# We exclude outliers from the data
data_new = data.loc[(data['Organism']!='Sphere 002') & (data['Organism']!='Sphere011')]

from lmfit import minimize, Parameters

def stokes_law_residual(params, diameter, data, eps_data):
    
    g = params['g']
    delta_rho = params['delta_rho']
    mu = params['mu']
    
    model = (diameter**2)*delta_rho*g/(18*mu)
    
    return (data-model)


params = Parameters()

params.add('g', value=9.81, vary = False)
params.add('mu', value=8.89e-4, vary = False)
params.add('delta_rho', value=1, vary = True)


# Load data into arrays (in SI units)
Diameters = (1e-3)*data_new['OrgSize_mean'].to_numpy()
SinkingSpeed_data = (1e-3)*data_new['v_Z'].to_numpy()

SinkingSpeed_standard_error =  (1e-3)*data_new['sigma_v_Z'].to_numpy()

print(SinkingSpeed_data)
print(SinkingSpeed_standard_error)



out = minimize(stokes_law_residual, params, args = (Diameters, SinkingSpeed_data, SinkingSpeed_standard_error**2))

# Print the results
print('Best fit density: {} kg/m^3'.format(out.params['delta_rho'].value))

# Calculate theoretical sinking speed based on best-fit Stokes-law
Size_array = np.linspace(0.7, 1.6, 20)*1e-3


TheoreticalSinkingSpeed = (Size_array**2)*out.params['delta_rho'].value*params['g']/(18*params['mu'])

print(TheoreticalSinkingSpeed)

# Plot the Mean sinking speed vs Size of sphere, Colored by No:of vorticella
title = 'Sinking speed vs size of model aggregates'

plt.figure(figsize=(16,12))


for ii in range(len(data)):
    print(ii)
    Organism = data['Organism'][ii]
    Condition = data['Condition'][ii]
    
    
    print(Organism)
    print(Condition)
    plt.errorbar(1000*data['OrgSize_mean'][ii], 1000*data['v_Z'][ii], xerr = 1000*data['OrgSize_std'][ii], yerr = 1000*data['sigma_v_Z'][ii], color = ColorStyle[Condition], marker = MarkerStyle[Condition], MarkerSize = 20, label = Condition, capsize = 5, linewidth=2, elinewidth=2, alpha=0.95, markeredgecolor = 'k')
    plt.annotate(Organism,(1000*data['OrgSize_mean'][ii],1000*data['v_Z'][ii]), textcoords="offset points", xytext=(20,20), ha='center', fontsize=12)

plt.plot(1e6*Size_array, 1e6*TheoreticalSinkingSpeed, 'k-', linestyle='--')
plt.xlabel('Sphere diameter (um)')
plt.ylabel('Sinking speed (um/s)')
plt.title(title)
    
from collections import OrderedDict

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
#plt.legend(loc=2,prop={'size': 30})
plt.savefig(os.path.join(figures_folder, title+'_StokesLawFit.png'), dpi =300)
plt.savefig(os.path.join(figures_folder, title+'_StokesLawFit.svg'), dpi =300)


plt.show()


#title = 'Peclet number of model aggregates'
#
#plt.figure(figsize=(16,12))
#
#
#for ii in range(len(data)):
#    print(ii)
#    Organism = data['Organism'][ii]
#    Condition = data['Condition'][ii]
#    
#    
#    print(Organism)
#    print(Condition)
#    plt.errorbar(1000*data['OrgSize_mean'][ii], 1000*data['v_Z'][ii], xerr = 1000*data['OrgSize_std'][ii], yerr = 1000*data['sigma_v_Z'][ii], color = ColorStyle[Condition], marker = MarkerStyle[Condition], MarkerSize = 20, label = Condition, capsize = 5, linewidth=2, elinewidth=2, alpha=0.95, markeredgecolor = 'k')
#    plt.annotate(Organism,(1000*data['OrgSize_mean'][ii],1000*data['v_Z'][ii]), textcoords="offset points", xytext=(20,20), ha='center', fontsize=12)
#
#plt.plot(1e6*Size_array, 1e6*TheoreticalSinkingSpeed, 'k-', linestyle='--')
#plt.xlabel('Sphere diameter (um)')
#plt.ylabel('Sinking speed (um/s)')
#plt.title(title)
#    
#from collections import OrderedDict
#
#handles, labels = plt.gca().get_legend_handles_labels()
#by_label = OrderedDict(zip(labels, handles))
#plt.legend(by_label.values(), by_label.keys())
##plt.legend(loc=2,prop={'size': 30})
#plt.savefig(os.path.join(figures_folder, title+'_StokesLawFit.png'), dpi =300)
#plt.savefig(os.path.join(figures_folder, title+'_StokesLawFit.svg'), dpi =300)
#
#
#plt.show()


