# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 13:02:49 2018

@author: Francois
"""

import csv as csv
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.stats as stats
import cmocean as cmocean
import os
import pandas as pd

#--------------------------------------------------------------------------
#                       Plots parameters
#--------------------------------------------------------------------------
from matplotlib import rcParams
from matplotlib import rc

rc('font', family='sans-serif') 
rc('font', serif='Helvetica') 
rc('text', usetex='false') 
rcParams.update({'font.size': 16})

#--------------------------------------------------------------------------
#                       Data importation
#--------------------------------------------------------------------------

path = '/Volumes/GRAVMACH1/HopkinsEmbroyologyCourse_GoodData/2018_06_12/Starfish/StarFish6highfreq'
file="track.csv"

dataFolder, TrackName = os.path.split(path)

#Test6_0_0_8mm_movTest2_0_2mm_away
Data=[]
reader = csv.reader(open(os.path.join(path, file),newline=''))
for row in reader:
    Data.append(row)
n=len(Data)

Time=np.array([float(Data[i][0])-float(Data[1][0]) for i in range(1,n)])             # Time stored is in milliseconds
Xobjet=np.array([float(Data[i][1]) for i in range(1,n)])             # Xpos in motor full-steps
Yobjet=np.array([float(Data[i][2]) for i in range(1,n)])             # Ypos in motor full-steps
Zobjet=np.array([float(Data[i][3]) for i in range(1,n)])             # Zpos is in encoder units
ThetaWheel=np.array([float(Data[i][4]) for i in range(1,n)])
ZobjWheel=np.array([float(Data[i][5]) for i in range(1,n)])
ManualTracking=np.array([int(Data[i][6]) for i in range(1,n)])   # 0 for auto, 1 for manual
ImageName=np.array([Data[i][7] for i in range(1,n)])
focusMeasure=np.array([float(Data[i][8]) for i in range(1,n)])
focusPhase=np.array([float(Data[i][9]) for i in range(1,n)])
MaxfocusMeasure=np.array([float(Data[i][10]) for i in range(1,n)])
#colorR=np.array([int(Data[i][11]) for i in range(1,n)])
#colorG=np.array([int(Data[i][12]) for i in range(1,n)])
#colorB=np.array([int(Data[i][13]) for i in range(1,n)])

#--------------------------------------------------------------------------
#                              Duration
#--------------------------------------------------------------------------

duration=' ( '+str(int(round(Time[-1]/60)))+' mn '+str(int(round(Time[-1]%60)))+' s )'
#--------------------------------------------------------------------------
#                       Boundaries equilibration
#--------------------------------------------------------------------------
xmin=Xobjet.min()
xmax=Xobjet.max()

ymin=Yobjet.min()
ymax=Yobjet.max()

print(ymin,ymax)

zmin=ZobjWheel.min()
zmax=ZobjWheel.max()

if xmax-xmin>15 and (xmin<-7.5 or xmax>7.5):
    delta_x=-np.mean(Xobjet)
    Xobjet=Xobjet+delta_x
    xmin+=delta_x
    xmax+=delta_x

elif xmin<-7.5:
    delta_x=-7.5-xmin
    Xobjet=Xobjet+delta_x
    xmin+=delta_x
    xmax+=delta_x
    
elif xmax>7.5:
    delta_x=7.5-xmax
    Xobjet=Xobjet+delta_x
    xmin+=delta_x
    xmax+=delta_x

if ymax-ymin>3 and (ymin<0 or ymax>3):
    delta_y=-(np.mean(Yobjet)-1.5)
    Yobjet=Yobjet+delta_y
    ymin+=delta_y
    ymax+=delta_y
    
elif ymin<0:
    delta_y=-ymin
    Yobjet=Yobjet+delta_y
    ymin+=delta_y
    ymax+=delta_y
    
elif ymax>3:
    delta_y=3-ymax
    Yobjet=Yobjet+delta_y
    ymin+=delta_y
    ymax+=delta_y
    
    
"""
#--------------------------------------------------------------------------
#                           Trajectory function of time
#--------------------------------------------------------------------------
xrange=int(np.ceil(xmax)-np.floor(xmin)+2)
yrange=int(np.ceil(ymax)-np.floor(ymin)+2)
zrange=int(np.ceil(zmax)-np.floor(zmin)+2)

plt.figure(figsize=plt.figaspect(1)*2)
#grid = plt.GridSpec(xrange+yrange+zrange, 1, wspace=1, hspace=8)
#
#ax7=plt.subplot(grid[xrange+yrange:])
ax7=plt.subplot(3,1,3)
plt.plot(Time,ZobjWheel)
plt.xlabel('T (s)')
plt.ylabel('Z (mm)')
plt.xlim( Time[0],Time[-1])
plt.ylim( np.floor(zmin)-1,np.ceil(zmax)+1)

#ax5=plt.subplot(grid[:xrange],sharex=ax7)
ax5=plt.subplot(3,1,1,sharex=ax7)
plt.plot(Time,Xobjet)
plt.ylabel('X (mm)')
p5 = patches.Rectangle((Time[0]-3, -7.5), Time[-1]-Time[0]+6, 15,fill=False,linestyle='--')
ax5.add_patch(p5)
plt.xlim( Time[0],Time[-1])
plt.ylim( np.floor(xmin)-1,np.ceil(xmax)+1)
plt.setp(ax5.get_xticklabels(), visible=False)



#ax6=plt.subplot(grid[xrange:xrange+yrange],sharex=ax7)
ax6=plt.subplot(3,1,2,sharex=ax7)
plt.plot(Time,Yobjet)
p6 = patches.Rectangle((Time[0]-3, 0), Time[-1]-Time[0]+6, 3,fill=False,linestyle='--')
ax6.add_patch(p6)
plt.xlim( Time[0],Time[-1])
plt.ylabel('Y (mm)')
plt.setp(ax6.get_xticklabels(), visible=False)

plt.ylim( np.floor(ymin)-1,np.ceil(ymax)+1)




plt.savefig(path+'Trajectory_in_Time.svg')

plt.show()

#--------------------------------------------------------------------------
#                           Zoom on Z axis
#--------------------------------------------------------------------------
T=[]
Z=[]
Timage=[]
Zimage=[]

for i in range(len(Time)):
    if Time[i]>186 and Time[i]<196:
        T.append(Time[i])
        Z.append(ZobjWheel[i])
        if len(ImageName[i])>3:
            Timage.append(Time[i])
            Zimage.append(ZobjWheel[i])

print(len(Zimage),len(Z))
plt.figure()
plt.plot(T,Z)
plt.scatter(Timage,Zimage,color='r',marker='x')
plt.xlim( T[0],T[-1])
plt.ylabel('Z (mm)')
plt.ylabel('t (s)')

plt.savefig(path+'blink_189.svg')

#--------------------------------------------------------------------------
#                       XZ for TimeLaps
#--------------------------------------------------------------------------

def discrete_cmap(N, cmap=None):
    #Create an N-bin discrete colormap from the specified input map

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:
    base=cmap
    colors=cmap(np.arange(256))
    colors=colors[50:]
    cmap=mpl.colors.ListedColormap(colors)
    color_list = cmap(np.linspace(0, 1, N))
    return base.from_list('deep_tronc', color_list, N)




selected_images=[]
Ximage2=[]
Zimage2=[]
Timage2=[]
for i in range(1730,1775,5):
    selected_images.append('IMG_'+str(i)+'.tif')
    index=list(ImageName).index(selected_images[-1])
    Ximage2.append(Xobjet[index])
    Zimage2.append(ZobjWheel[index])
    Timage2.append(Time[index])
    
selected_images1=[]
Ximage1=[]
Zimage1=[]
Timage1=[]
for i in range(1730,1775,1):
    selected_images.append('IMG_'+str(i)+'.tif')
    index=list(ImageName).index(selected_images[-1])
    Ximage1.append(Xobjet[index])
    Zimage1.append(ZobjWheel[index])
    Timage1.append(Time[index])
    

zmax2=round(max(Zimage2),1)+0.3
zmin2=round(min(Zimage2),1)-0.3

xmax2=round(max(Ximage2),1)+0.3
xmin2=round(min(Ximage2),1)-0.3

ticks=[round(i,1) for i in Timage2 ]

print(ticks)

plt.figure()
N=len(Timage2)


plt.plot(Ximage1,Zimage1)
plt.scatter(Ximage2,Zimage2,c=Timage2,cmap= discrete_cmap(N, cmocean.cm.deep), marker='o',s=0.6)
plt.colorbar(ticks=ticks)

plt.xlabel('X (mm)')
plt.ylabel('Z (mm)')



plt.axis('equal')

plt.xlim(xmin2,xmax2)
plt.ylim(zmin2,zmax2)

#cbar3=fig3.colorbar(sc3,ticks=Timage2)
#cbar3.set_label('Time (s)')
plt.savefig(path+'XZ_Trajectory.svg',dpi=300)
"""
#--------------------------------------------------------------------------
#                       Frequency of the blinks
#--------------------------------------------------------------------------

def rot(T,Z,theta):
    alpha=np.arctan(Z/T)
    norme=(Z**2+T**2)**0.5
    return norme*np.cos(alpha-theta),norme*np.sin(alpha-theta)

#T=[Time[0]]
#Z=[ZobjWheel[0]]
#for i in range(1,len(Time)):
#    t,z=rot(Time[i],ZobjWheel[i],np.pi/12)
#    T.append(t)
#    Z.append(z)
#    
#    
#f = plt.figure()
#plt.plot(T,Z)
#plt.show()
 
avgWindow = 500

Z_movAvg = pd.rolling_mean(ZobjWheel, avgWindow, min_periods = 1, center = True)

Z_fast = ZobjWheel - Z_movAvg    

f = plt.figure()
plt.plot(Time, Z_fast)
plt.show()



peaks = find_peaks(Time,Z_fast)

print(peaks)
DeltaTBlink=[]
for i in range(1,len(peaks)-1):
    DeltaTBlink.append(Time[peaks[i+1]]-Time[peaks[i]])
    

DeltaTBlink=np.array(DeltaTBlink)
freq=np.mean(1/DeltaTBlink)
std=np.std(1/DeltaTBlink)

print('freq',freq,'std',std)

peak_indicator=[0 for i in range(len(Time))]

for j in peaks:
    peak_indicator[j]= 1

plt.figure(figsize=(8,6))

indexArray = np.array([i for i in range(len(Time))],dtype='int')

plt.plot(Time, ZobjWheel,color='k',label='trajectory')

#for ii in indexArray[peak_indicator]:
#    plt.vlines(Time[ii],np.floor(ZobjWheel.min())-1,ZobjWheel[ii],linewidth=1,linestyles='-',alpha=0.5,color='r')
#plt.plot(Time, Z_movAvg, color='r')
plt.scatter(Time[peak_indicator],ZobjWheel[peak_indicator], 5, color='r',label='blink')
plt.title('"Blink" detection')
plt.legend()
plt.xlim( Time[0],Time[-1])
plt.savefig(TrackName+'_Blink_Detection.svg')
plt.ylim( np.floor(ZobjWheel.min())-1,np.ceil(ZobjWheel.max()+1))
plt.show()


plt.figure()
weights = np.ones_like(DeltaTBlink)/float(len(DeltaTBlink))
plt.hist(DeltaTBlink,bins=50)
plt.title('StarFish10: "Blink" distribution')
plt.xlabel('Delta T (s)')
plt.ylabel('nb of blinks')
plt.savefig(TrackName+'_Blink_Distribution.svg')


