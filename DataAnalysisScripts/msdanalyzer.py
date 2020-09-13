# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 14:45:49 2018
Mean-Squared-Displacement Analyzer for multiple tracks
@author: deepak90
"""
import numpy as np
# import Track
import os
import imp
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
from cycler import cycler
import pandas as pd
import GravityMachineTrack
import PlotFunctions.PlotUtils as PlotUtils
imp.reload(GravityMachineTrack)
from wlsice.python import wlsice as wlsice
# imp.reload(wlsice)
import seaborn as sns
from pyentrp import entropy as ent
from datetime import datetime

def squaredDisp(data):
    
    return (data - data[0])**2

class msdanalyzer:
    
    def __init__(self, Tracks = None, nDims = 3, ensemble_method = 'subtrack', testFlag = False, v = None, tau = None, Organism = None, Condition = None, savePath = None):
        # Tracks is a list of gravity machine tracks
        self.precision = 12
        self.nDims = 3
        self.testFlag = testFlag
        dateTimeObj = datetime.now()
        if(self.testFlag):
            self.Organism = 'Run_Tumble_Swimmers'
            
            self.Condition = dateTimeObj.strftime("%d-%b-%Y_%H_%M_%S")

        else:
            self.Organism = Organism
            self.Condition = Condition
            
        self.rootFolder = savePath

        self.figuresFolder = os.path.join(self.rootFolder, 'MSD_figures')

        self.velocityDistFolder = os.path.join(self.rootFolder, 'VelocityTimeSeries')



        if(not os.path.exists(self.figuresFolder)):
            os.makedirs(self.figuresFolder)

        self.fittingFolder = os.path.join(self.rootFolder, 'TaylorFunctionFitting_Analysis' +dateTimeObj.strftime("%d-%b-%Y"))

        if(not os.path.exists(self.fittingFolder)):
            os.makedirs(self.fittingFolder)

        if(not os.path.exists(self.velocityDistFolder)):
            os.makedirs(self.velocityDistFolder)
            
#        self.entropyFolder = os.path.join(self.rootFolder, 'MSE_calculation')
#
#        if(not os.path.exists(self.entropyFolder)):
#            os.makedirs(self.entropyFolder)

        # Path in which to save the resulting MSD trajectories
        self.saveFile = str(self.Organism)+'_'+str(self.Condition)
        # Create sub-folder for each condition
        self.subFolder = str(self.Organism) + '_' + str(self.Condition)

        # Path to the sub-folder which contains the trajectories
        self.savePath = os.path.join(self.rootFolder, self.subFolder)


        if(not os.path.exists(self.savePath)):
            os.makedirs(self.savePath)

        self.velocities_df = None

        # Two possible ways of calculating ensembles: Over tracks broken into subtracks with a specific max Delay, or over individual tracks with max delay = min (track durations)
        # subtrack OR mintrack
        self.ensemble_method = ensemble_method

        if(self.testFlag == 1 and Tracks is None):
            
        
            self.maxDelay = 50
            self.timeWindow = 5
            # Length of the delays array for Sub-sampling of the trajectory and over which we calculate MSD
            self.tLen = 1000
            # self.genTestTraj()
            self.genRunTumbleTraj(v = v, tau = tau)
            
        else:
            self.maxDelay = 50
            self.timeWindow = 5
            if(Tracks is not None):
                self.Tracks = Tracks
                self.nTracks = len(Tracks)
                
                print(self.nTracks)
                 # Maximum delay over which we want to calculate correlations
                
                self.getMinDiff()

                if(ensemble_method == 'mintrack'):
                    # Calculate the min track duration
                    self.getMinTrackDuration()


                self.getTotalTrackDuration()
                self.getCommonDelays()
                
                self.init_entropy_params()



    def get_Mean_Std(self, attr):
        # Calculate mean and std of attribute over a group of tracks

        Array = []

        for ii in range(self.nTracks):

            currTrack = self.Tracks[ii]

            Array.append(getattr(currTrack, attr))

        return np.nanmean(Array), np.nanstd(Array)


       
        
    def genTestTraj(self):
        
        Tracks = []
        
        self.nTracks = 100
        
        boxSize = 2
        
        timeSteps = 1000
        self.DiffCoeff = 100
        dT = 0.05
        
        k = (self.DiffCoeff*dT)**(1/2)
        
        TimeArray = np.array(range(0,timeSteps))*dT
        
        for ii in range(self.nTracks):
            
            
            X0 = boxSize*np.random.rand(self.nDims)
            dX = k*np.random.randn(self.nDims,timeSteps)
            
            dX[:,0] = X0;
            
            X = np.cumsum(dX,1)
            
            Tracks.append(GravityMachineTrack.gravMachineTrack())
            Tracks[ii].T = TimeArray
            Tracks[ii].X = X[0,:]
            Tracks[ii].Y = X[1,:]
            Tracks[ii].ZobjWheel = X[2,:]
            
        
        self.Tracks = Tracks
        
        # Common time array for Sub-sampling of the trajectory.
        self.delays = np.linspace(0,self.maxDelay,self.tLen)
#        print(Tracks[0].T)
            
        
    def genRunTumbleTraj(self, v = 1, tau = 1):

        Tracks = []
        
        self.nTracks = 50

        print('Generating test trajectories for {} swimmers \n with velocity {} and run-time {}'. format(self.nTracks, v, tau))

        
        timeSteps = 1000

        boxSize = 2
        
        # Time steps for the simulations
        dT = 0.1*tau

        # ballistic velocity of the swimmers
        self.velocity = v

        # Mean rate of tumbles is the inverse of the run-time tau
        self.tumble_rate = 1/tau

        # Initial particle positions
        X0 = boxSize*np.random.rand(self.nDims, self.nTracks)

        # Array to store particle orientation
        P0 = np.zeros((self.nDims, self.nTracks))

        # Arrays to store the time series data
        X_array = np.zeros((self.nDims, self.nTracks, timeSteps))
        
        TimeArray = np.array(range(0,timeSteps))*dT

        # Array to store the orientations of all the swimmers
        P_array = np.zeros_like(P0)

        rand_array =np.random.rand(self.nDims - 1, self.nTracks)

        phi = 2*np.pi*rand_array[0,:]

        if(self.nDims>2):
            q = 2*rand_array[1,:]-1;
        else:
            q = np.zeros((1,self.nTracks))

        cos_theta = q
        sin_theta = (1-q**2)**(1/2)
       
        P0[0,:] = sin_theta*np.cos(phi)
        P0[1,:] = sin_theta*np.sin(phi)
        P0[2,:] = cos_theta

        pdot = np.sum(P0*P0, axis = 0)**(1/2)

        P0[0,:] = P0[0,:]/pdot
        P0[1,:] = P0[1,:]/pdot
        P0[2,:] = P0[2,:]/pdot

        P_array = P0
        X_array[:,:,0] = X0

        for ii in range(timeSteps-1):


            tumble_rand = np.array(np.random.rand(self.nTracks) <= self.tumble_rate*dT, dtype = 'bool')

          

            N_tumble = np.sum(tumble_rand)
            print(N_tumble)

            rand_array =np.random.rand(self.nDims - 1, N_tumble)

            phi = 2*np.pi*rand_array[0,:]

            if(self.nDims>2):
                q = 2*rand_array[1,:]-1
            else:
                q = np.zeros((1,N_tumble))

            cos_theta = q
            sin_theta = (1-q**2)**(1/2)
            
            P_new = np.zeros((self.nDims, N_tumble))

            P_new[0,:] = sin_theta*np.cos(phi)
            P_new[1,:] = sin_theta*np.sin(phi)
            P_new[2,:] = cos_theta

            

            # Update the orientations of particles undergoing a tumble
            
            P_array[:, tumble_rand] = P_new

            pdot = np.sum(P_array*P_array, axis = 0)**(1/2)

            # print(pdot)

            P_array[0,:] = P_array[0,:]/pdot
            P_array[1,:] = P_array[1,:]/pdot
            P_array[2,:] = P_array[2,:]/pdot



            X_array[:,:,ii+1] = X_array[:,:,ii] + self.velocity*P_array*dT


        for ii in range(self.nTracks):

            Tracks.append(GravityMachineTrack.gravMachineTrack())
            Tracks[ii].T = TimeArray
            Tracks[ii].X = X_array[0,ii,:]
            Tracks[ii].Y = X_array[1,ii,:]
            Tracks[ii].ZobjWheel = X_array[2,ii,:]

            if(ii==1):
                plt.figure()
                plt.plot(Tracks[ii].T, Tracks[ii].X,'k')
                plt.show()



        self.Tracks = Tracks
        
        self.delays = np.linspace(0,self.maxDelay,self.tLen)


    def getTotalTrackDuration(self):

        self.totalTrackDuration = 0

        for ii in range(self.nTracks):
            currTrack = self.Tracks[ii]

            self.totalTrackDuration += currTrack.trackDuration


        print('Total track duration : {} s'.format(self.totalTrackDuration))


    def getMinTrackDuration(self):

        self.minTrackDuration = 1000000000
        for ii in range(self.nTracks):
            currTrack = self.Tracks[ii]

            self.minTrackDuration = min(self.minTrackDuration, currTrack.trackDuration)


        print('Min track duration : {} s'.format(self.minTrackDuration))
        self.maxDelay = self.minTrackDuration



    def getMinDiff(self):
        
        
        minDiff = 10
        maxDiff = 0
        for ii in range(self.nTracks):
            TimeArray = self.Tracks[ii].T
#            time = np.hstack((time, TimeArray))
            
            diff = np.diff(TimeArray)
#            print('Track {}, Time Array : {}'.format(ii,TimeArray))
            
            minDiff = min(np.min(diff), minDiff)
            maxDiff = max(np.max(diff), maxDiff)
           
        
#        time = np.unique(time)
        
#        print(np.shape(time))
        
#        time = np.round(time, decimals = self.precision)
        
#        self.time = time
        self.minDiff = minDiff
        self.maxDiff = maxDiff
        
        print(self.minDiff)
        print(self.maxDiff)
        
    def getCommonDelays(self):
        # This generates a common time array so many different tracks with different sampling rates can be avearaged
        
        self.tLen = int(self.maxDelay/float(self.minDiff))
        self.delays = np.linspace(0,self.maxDelay,self.tLen)
        self.delayIndex = np.array(range(0,len(self.delays)))
#        print(self.delays)
        
    def DispAtCommonTimes(self):
        
        Disp_X = []
        Disp_Y = []
        Disp_Z = []
        
       
        maxTrackLen = 0
        
        for ii in range(self.nTracks):


            currTrack = self.Tracks[ii]
            
            TimeArray = np.arange(0,currTrack.T[-1],self.minDiff)
            
            print('...Analyzing Track {} of Duration {} s'.format(ii, np.max(currTrack.T)))

            TimeArray = currTrack.T
            
            func_X = interpolate.interp1d(currTrack.T,currTrack.X, kind = 'linear')
            func_Y = interpolate.interp1d(currTrack.T,currTrack.Y, kind = 'linear')
            func_Z = interpolate.interp1d(currTrack.T,currTrack.ZobjWheel, kind = 'linear')
            
            Disp_X.append(func_X(TimeArray))
            Disp_Y.append(func_Y(TimeArray))
            Disp_Z.append(func_Z(TimeArray))
            
            maxTrackLen = max(maxTrackLen, len(TimeArray))
            
        return maxTrackLen, Disp_X, Disp_Y, Disp_Z
    
    
    def computeMeanDisp(self):
        maxTrackLen, Disp_X, Disp_Y, Disp_Z = self.DispAtCommonTimes()
        
        nSubTracks = len(Disp_X)
        
        print('Total number of subtracks : {}'.format(nSubTracks))
        
        Disp_matrix_X = np.zeros((nSubTracks,maxTrackLen))
        Disp_matrix_Y = np.zeros((nSubTracks,maxTrackLen))
        Disp_matrix_Z = np.zeros((nSubTracks,maxTrackLen))
        
#        print(np.shape(MSD_matrix_X))
        
       
        for ii in range(nSubTracks):
            
#            print(np.shape(SquaredDisp_X[ii]))
            Disp_matrix_X[ii,:len(Disp_X[ii])] =  Disp_X[ii]
            Disp_matrix_Y[ii,:len(Disp_Y[ii])] =  Disp_Y[ii]
            Disp_matrix_Z[ii,:len(Disp_Z[ii])] =  Disp_Z[ii]
            
        
        Weights = np.sum(Disp_matrix_X!=0, axis = 0)
            
        Disp_matrix_X[Disp_matrix_X==0] = np.nan
        Disp_matrix_Y[Disp_matrix_Y==0] = np.nan
        Disp_matrix_Z[Disp_matrix_Z==0] = np.nan
        
        
        
        
        
        
        self.meanDisp_X = np.nanmean(Disp_matrix_X,axis = 0)
        self.meanDisp_Y = np.nanmean(Disp_matrix_Y,axis = 0)
        self.meanDisp_Z = np.nanmean(Disp_matrix_Z,axis = 0)
        
        self.meanDisp_XY = np.nanmean(Disp_matrix_X + Disp_matrix_Y, axis = 0)
        
        self.stdDisp_X = np.nanstd(Disp_matrix_X,axis = 0)
        self.stdDisp_Y = np.nanstd(Disp_matrix_Y,axis = 0)
        self.stdDisp_Z = np.nanstd(Disp_matrix_Z,axis = 0)
        
        self.stdDisp_XY = np.nanstd(Disp_matrix_X + Disp_matrix_Y, axis = 0)
        
    def computeSqDisp_minTrack(self, save = False):
        # Squared displacements where the MaxDelay is set by the min Track length
        SquaredDisp_X = []
        SquaredDisp_Y = []
        SquaredDisp_Z = []
        
        
        for ii in range(self.nTracks):
            
            
            currTrack = self.Tracks[ii]
        
            print('...Analyzing Track {} of Duration {} s'.format(ii, np.max(currTrack.T)))
            TimeArray = currTrack.T
            
            func_X = interpolate.interp1d(currTrack.T,currTrack.X, kind = 'linear')
            func_Y = interpolate.interp1d(currTrack.T,currTrack.Y, kind = 'linear')
            func_Z = interpolate.interp1d(currTrack.T,currTrack.ZobjWheel, kind = 'linear')


            X_subTrack = func_X(self.delays)
            Y_subTrack = func_Y(self.delays)
            Z_subTrack = func_Z(self.delays)
                
            # if(save_trajectories==True):
            #     # Save the trajectories
                
                
            SquaredDisp_X.append(squaredDisp(X_subTrack))
            SquaredDisp_Y.append(squaredDisp(Y_subTrack))
            SquaredDisp_Z.append(squaredDisp(Z_subTrack))


        return SquaredDisp_X, SquaredDisp_Y, SquaredDisp_Z


            
            

    
    def computeSqDisp(self, save = False):
        
        print('Computing squared displacements ...')

        # if(not os.path.exists(self.savePath) or len(os.listdir(self.savePath))==0): 
        
        SquaredDisp_X = []
        SquaredDisp_Y = []
        SquaredDisp_Z = []
        
        counter = 0
        for ii in range(self.nTracks):
            
            
            currTrack = self.Tracks[ii]
        
            print('...Analyzing Track {} of Duration {} s'.format(ii, np.max(currTrack.T)))
            TimeArray = currTrack.T
            
            func_X = interpolate.interp1d(currTrack.T,currTrack.X, kind = 'linear')
            func_Y = interpolate.interp1d(currTrack.T,currTrack.Y, kind = 'linear')
            func_Z = interpolate.interp1d(currTrack.T,currTrack.ZobjWheel, kind = 'linear')
            
            
            
            
            T_start = TimeArray[0]
            startIndex = 0
            
            while (T_start < np.max(TimeArray)-self.maxDelay):
                
                T_subTrack = T_start + self.delays
                
                
                X_subTrack = func_X(T_subTrack)
                Y_subTrack = func_Y(T_subTrack)
                Z_subTrack = func_Z(T_subTrack)
                
                SqDisp_X_subTrack = squaredDisp(X_subTrack)
                SqDisp_Y_subTrack = squaredDisp(Y_subTrack)
                SqDisp_Z_subTrack = squaredDisp(Z_subTrack)
                
                SquaredDisp_X.append(SqDisp_X_subTrack)
                SquaredDisp_Y.append(SqDisp_Y_subTrack)
                SquaredDisp_Z.append(SqDisp_Z_subTrack)

                # Save the Squared displacements for each trajectory
                if(save is True):
#                    print('Saving trajectory {}'.format(counter))
                    np.savez_compressed(os.path.join(self.savePath,self.Organism + '_'+str(self.Condition) + '_' + "trajectories"+'{:04d}'.format(counter)), trajectories_x = SqDisp_X_subTrack, trajectories_y = SqDisp_Y_subTrack, trajectories_z = SqDisp_Z_subTrack, time = T_subTrack)


                startIndex += next((i for i,x in enumerate(TimeArray[startIndex:] - TimeArray[startIndex]) if x >= self.timeWindow), None)
                
                T_start = TimeArray[startIndex]
                
                counter += 1
                
            print('no:of Subtracks from Track {}: {}'.format(ii, counter))
        
        return SquaredDisp_X, SquaredDisp_Y, SquaredDisp_Z



    
    
    def computeMSD(self, save = False, overwrite = False):
        
        saveFile = self.saveFile + '_MSD.csv'

        if(not os.path.exists(os.path.join(self.rootFolder, saveFile)) or overwrite):

            if(self.ensemble_method == 'subtrack'):
                SquaredDisp_X, SquaredDisp_Y, SquaredDisp_Z = self.computeSqDisp(save = True)
            else:
                SquaredDisp_X, SquaredDisp_Y, SquaredDisp_Z = self.computeSqDisp_minTrack()

            
            # Find the no:of subtracks. This should be the same as number of tracks if we use the mintrack ensemble method.
            nSubTracks = len(SquaredDisp_X)
            
            print('Total number of subtracks : {}'.format(nSubTracks))
            
            SD_matrix_X = np.zeros((nSubTracks,self.tLen))
            SD_matrix_Y = np.zeros((nSubTracks,self.tLen))
            SD_matrix_Z = np.zeros((nSubTracks,self.tLen))
            
    #        print(np.shape(MSD_matrix_X))
            
           
            for ii in range(nSubTracks):
                
    #            print(np.shape(SquaredDisp_X[ii]))
                SD_matrix_X[ii,:len(SquaredDisp_X[ii])] =  SquaredDisp_X[ii]
                SD_matrix_Y[ii,:len(SquaredDisp_Y[ii])] =  SquaredDisp_Y[ii]
                SD_matrix_Z[ii,:len(SquaredDisp_Z[ii])] =  SquaredDisp_Z[ii]
                
    #        subTracks = np.array(range(nSubTracks))
            
    #        print(len(subTracks))
    #        print(len(self.delays))
    #        print(np.shape(MSD_matrix_X))
    #        plt.figure(4)
    #        ax = plt.contourf(self.delays,subTracks,MSD_matrix_X==0)
    #        plt.colorbar(ax)
    #        plt.axis('image')
    #        plt.show()
                

            
            
            
            
            
            
            self.MSD_X = np.nanmean(SD_matrix_X,axis = 0)
            self.MSD_Y = np.nanmean(SD_matrix_Y,axis = 0)
            self.MSD_Z = np.nanmean(SD_matrix_Z,axis = 0)
            
            self.MSD_XY = np.nanmean(SD_matrix_X + SD_matrix_Y, axis = 0)

            self.MSD_3D = np.nanmean(SD_matrix_X + SD_matrix_Y + SD_matrix_Z, axis = 0)
            
            self.stdev_X = np.nanstd(SD_matrix_X,axis = 0)
            self.stdev_Y = np.nanstd(SD_matrix_Y,axis = 0)
            self.stdev_Z = np.nanstd(SD_matrix_Z,axis = 0)
            
            self.stdev_XY = np.nanstd(SD_matrix_X + SD_matrix_Y, axis = 0)

            self.stdev_3D = np.nanstd(SD_matrix_X + SD_matrix_Y + SD_matrix_Z, axis = 0)


            if(save is True):
                dataLen = len(self.MSD_X)

                self.OrgSize_mean, self.OrgSize_std = self.get_Mean_Std('OrgDim')

                print('Mean and Std of Organism size : {} +- {}(um)'.format(self.OrgSize_mean, self.OrgSize_std))

                # Save the results of the MSD analysis as numpy arrays/ pandas dataframes ...
                dataFrame = pd.DataFrame({'Organism':[], 'nTracks':[], 'Total duration':[], 'OrgSize mean':[],'OrgSize std':[],'Condition':[], 'delays':[],'MSD_X':[],'MSD_Y':[], 'MSD_Z':[], 'stdev_X':[], 'stdev_Y':[], 'stdev_Z':[]})

                dataFrame = dataFrame.append(pd.DataFrame({'Organism':np.repeat(self.Organism,dataLen,axis = 0), 'nTracks':np.repeat(self.nTracks,dataLen,axis = 0), 'Total duration':np.repeat(self.totalTrackDuration,dataLen,axis = 0), 'OrgSize mean':np.repeat(self.OrgSize_mean,dataLen,axis = 0), 'OrgSize std':np.repeat(self.OrgSize_std,dataLen,axis = 0), 'Condition':np.repeat(self.Condition,dataLen,axis = 0),'delays':self.delays,'MSD_X':self.MSD_X,'MSD_Y':self.MSD_Y, 'MSD_Z':self.MSD_Z, 'stdev_X': self.stdev_X, 'stdev_Y':self.stdev_Y, 'stdev_Z':self.stdev_Z}))
                
                dataFrame.to_csv(os.path.join(self.rootFolder, saveFile))

        else:
            print('Loading MSD trajectory from file ...')
            self.load_MSD(fileName = saveFile )



    def load_MSD(self, fileName = None):

        keys = ['delays', 'MSD_X','MSD_Y','MSD_Z','stdev_X','stdev_Y','stdev_Z']
        
        if(fileName is not None):
            print('Loading MSD data from {}'.format(os.path.join(self.rootFolder, fileName)))
            dataFrame = pd.read_csv(os.path.join(self.rootFolder, fileName))

            for attr in keys:

                setattr(self, attr, dataFrame[attr])

            self.OrgSize_mean, self.OrgSize_std = dataFrame['OrgSize mean'][0], dataFrame['OrgSize std'][0]
    
    
        
        
#    def computeLocalSlope(self, TimeArray, Track):
#
#        TimeArray = np.array(TimeArray)
#        Track = np.array(Track)
#
#        # Time window over which data is used for extracting the slope
#        window_size = int(self.maxDelay/10)
#        print('Window size {}(s)'.format(window_size))
#
#        # Time resolution of the extracted slope curve
#        step_size = (TimeArray[1] - TimeArray[0])
#
#        T_start = TimeArray[1]
#        startIndex = 0
#        counter = 0
#
#        Slope_Array = []
#        Delays_Array = []
#
#        while (T_start < np.max(TimeArray)-window_size):
#                
#            mask1 = TimeArray>=T_start 
#            mask2 = TimeArray<= T_start + window_size
#            T_subTrack = TimeArray[mask1 & mask2]
#
#            subTrack = Track[mask1 & mask2]
#
#            # Convert to log-scale to extract the slope
#            T_log = np.log(T_subTrack)
#            subTrack_log = np.log(subTrack)
#
#            # plt.figure()
#            # plt.plot(T_log, subTrack_log, 'ro')
#            # plt.show()
#
#            # Perform a linear fit and record the slope
#            p = np.polyfit(T_log, subTrack_log, deg = 1)
#
#            Slope_Array.append(p[0])
#            Delays_Array.append(T_start)
#
#            T_start += step_size
#
#
#        Slope_Array = np.array(Slope_Array)
#        Delays_Array = np.array(Delays_Array)



#        return Delays_Array, Slope_Array
    #--------------------------------------------------------------------------------------------------------
    # Weighted-Least-Squares with Correlated Errors Fitting Functions
    #--------------------------------------------------------------------------------------------------------
    # Define the power-law function, its gradient and Hessian. 

    def f_powerlaw(self, t, params):
        """A powerlaw: a*t**b"""
        assert(len(params) == 2)
        return np.multiply(params[0], np.power(t, params[1]))
    
    
    # np.array(1), np.array(1) -> np.array(2)
    def df_powerlaw(self, t, params):
        """Gradient of a powerlaw, with regards to parameters a,b:
        gradient(a*t**b) = [t**b, a*log(t)*t**b]
        """
        assert(t[0] > 0)
        assert(len(params) == 2)
        a = params[0]
        b = params[1]
    
        dfd_lam = np.zeros((len(params), len(t)))
        dfd_lam[0] = np.power(t, b) # list of N numbers
        dfd_lam[1] = np.multiply(a*np.log(t), np.power(t, b)) # list of N numbers
        return dfd_lam
    
    # np.array(1), np.array(1) -> np.array(3!)
    def d2f_powerlaw(self, t, params):
        """Return Hessian of powerlaw"""
        a = params[0]
        b = params[1]
    
        d2f = np.zeros((len(params), len(params), len(t)))
    
        # d2f / da da
        # df2[0,0] = np.zeroes(np.shape(t)) #(already there)
    
        # d2f / da db
        d2f[0,1] = np.multiply(np.log(t), np.power(t,b))
    
        # d2f / db da
        d2f[1,0] = d2f[0,1]
    
        # d2f / db db
        d2f[1,1] = np.multiply(a*(np.log(t)**2), np.power(t,b))
    
        return d2f
    
    
    # Ballistic motion with velocity v
    #--------------------------------------------------------------------------------------------------------
    def f_ballistic(self, t, params):

        assert(len(params) == 1)
        v = params[0]

        return v**2*t**2

    def df_ballistic(self, t, params):
        assert(len(params) == 1)
        v = params[0]
        
        df = np.zeros((len(params), len(t)))

        df[0,:] = 2*v*t**2

        return df

    def d2f_ballistic(self, t, params):
        assert(len(params) == 1)
        v = params[0]

        d2f = np.zeros((len(params), len(params), len(t)))

        d2f[0,0,:] = 2*t**2
        
        return d2f 
    #--------------------------------------------------------------------------------------------------------
    # Taylor function with velocity v and correlation time tau
    #--------------------------------------------------------------------------------------------------------
    def f(self, t, params):
        """ Taylor function:
        """ 
        assert(len(params) == 2)

        v = params[0]
        tau = params[1]

        return 2*v**2*tau*(t - tau*(1 - np.exp(-t/tau)))

    def df(self, t, params):
        """ Gradient of Taylor function with respect to parameters v, tau """

        assert(len(params) == 2)
        v = params[0]
        tau = params[1]

        df = np.zeros((len(params), len(t)))

        # Gradient wrt v (param[0])
        df[0] = 4*tau*(t+(-1+(np.e)**(-1*t*(tau)**(-1)))*tau)*v

        df[1] = 2*(t+(-2*tau+(np.e)**(-1*t*(tau)**(-1))*(t+2*tau)))*(v)**(2)

        return df

    def d2f(self, t, params):

        v = params[0]
        tau = params[1]

        d2f = np.zeros((len(params), len(params), len(t)))

        d2f[0,0] = 4 * tau * ( t + ( -1 + ( np.e )**( -1 * t * ( tau )**( -1 ) ) )*tau )

        d2f[0,1] = 4 * ( t + ( -2 * tau + ( np.e )**( -1 * t * ( tau )**( -1 ) ) * (t + 2 * tau ) ) ) * v

        d2f[1,0] = 4 * ( t + ( -2 * tau + ( np.e )**( -1 * t * ( tau )**( -1 ) ) * ( t + 2 * tau ) ) ) * v

        d2f[1,1] = 2 * ( -2 + ( np.e )**( -1 * t * ( tau )**( -1 ) ) * ( tau )**( -2 ) * ( ( t )**( 2 ) + ( 2 * t * tau + 2 * ( tau )**( 2 ) ) ) ) * ( v )**( 2 )

        return d2f

            

    def load_trajectories(self):
        # Load trajectories from a folder and calculate MSD, and also perform fitting to extract parameters
        
        TimeArray = []
        Traj_X = []
        Traj_Z = []

        counter = 0
        for file in os.listdir(self.savePath):

            if file.endswith(".npz"):
                Traj_X.append([])
                Traj_Z.append([])
#                print('Loading trajectory : {}'.format(file))
                data = np.load(os.path.join(self.savePath, file))
                Time = data['time'] - data['time'][0]

                # Save the time the first time around
                if(counter==0):
                    TimeArray.append(Time)
                    
#                print(len(data['trajectories_x']))
                Traj_X[counter].append(data['trajectories_x'])
                Traj_Z[counter].append(data['trajectories_z'])
                
                
                counter += 1
        
        print(Traj_X)
        TimeArray = np.squeeze(np.array(TimeArray))
        Traj_X = np.squeeze(np.array(np.stack(Traj_X, axis = 0)))
        Traj_Z = np.squeeze(np.array(np.stack(Traj_Z, axis = 0)))

        return TimeArray, Traj_X, Traj_Z

    def TruncateTrajectory(self, Time, Traj, Tmax):
        # Max time over which to calculate the fit for the X-axis trajectory
        Time, Traj = np.array(Time), np.array(Traj)

#        print(Time)

        print(np.shape(Time))
        print(np.shape(Traj))

        Tmax_index = next((i for i,x in enumerate(Time) if x >= Tmax))

        Tmax_index = Tmax_index

        print(Tmax_index)

        return Time[:Tmax_index], Traj[:, :Tmax_index]

    def coff_det(self, y, f):
        """Coefficient of determination, to check goodness-of-fit, from mean
        of data (MSD), and f(t, params).
        https://en.wikipedia.org/wiki/Coefficient_of_determination
        """
        N = len(y)

        # note, by y_mean here we mean the average over N, not M!
        y_mean = np.sum(y) / N

        SS_tot = np.sum(np.square(np.subtract(y_mean, y)))
        SS_reg = np.sum(np.square(np.subtract(y_mean, f)))
        SS_res = np.sum(np.square(np.subtract(y,      f)))

        coeff_det = (1- SS_res / SS_tot)

        print("# Coefficient of determination:\t {}".format(coeff_det))

        return coeff_det


    def performWLSICEfit(self, Time, Traj, function = 'taylor'):

        


        M, N = np.shape(Traj)
        
        print('No:trajectories: {}, No:of time points {}'.format(M,N))

        min_method = "nm"           # Use Nelder-Mead minimization method
        
        

        print('Initiaizing WLS-ICE fitting with {}'.format(function))
        # The analytical function to fit, its gradient, and hessian
        if(function == 'taylor'):
            guess=np.array([1.0,1.0])     # Starting point in parameter space
            wlsice.init(self.f, self.df, self.d2f)

        elif (function == 'ballistic'):
            guess=np.array([1.0])     # Starting point in parameter space
            wlsice.init(self.f_ballistic, self.df_ballistic, self.d2f_ballistic)


        print('Performing the fit ...')

         # Perform the actual fit
        params, sigma, chi2_min = wlsice.fit(Time, Traj, guess, min_method)

          # RESULT:
        print("# trajectories M = {},\tsampling times N={}, t_0={}".format(M, N, Time[0]))
        print("# Optimal param:\t {}".format(abs(params)))
        print("# Sigma:\t {}".format(sigma))
        print("# Chi-square value: \t {}".format(chi2_min))

        # Also get goodness-of-fitt parametes
        y_mean = wlsice.computeMean(Traj)

        if(function == 'taylor'):
            coeff_det = self.coff_det(y_mean, self.f(Time, params))
        elif (function == 'ballistic'):
            coeff_det = self.coff_det(y_mean, self.f_ballistic(Time, params))
            
        

        return params, sigma, chi2_min, coeff_det


    def fitTrajectories(self, overwrite = False):

        saveFile = self.saveFile + '_TaylorFunctionFit.csv'

        if(not os.path.exists(os.path.join(self.fittingFolder, saveFile)) or overwrite == True):

            Time, Traj_X, Traj_Z = self.load_trajectories()

            Time = Time[1:]
            Traj_X, Traj_Z = Traj_X[:,1:], Traj_Z[:,1:]

            shape = np.shape(Traj_X)
#            print(shape)
#            print(Traj_X)
            assert(shape[1]>1)
            # Truncate the X trajectory over a shorter time, since it typically decorrelates faster
            print(len(Time))
            Time_X, Traj_X = self.TruncateTrajectory(Time, Traj_X, 10)
            # Time_X = Time
            # shape = np.shape(Traj_X)
            # print(shape)

            self.params_X, self.sigma_X, self.chi2_min_X, self.coeff_det_X = self.performWLSICEfit(Time_X, Traj_X, function = 'taylor')

            self.params_Z, self.sigma_Z, self.chi2_min_Z, self.coeff_det_Z = self.performWLSICEfit(Time, Traj_Z, function = 'ballistic')


            self.v_X, self.tau_X = abs(self.params_X[0]), self.params_X[1]
            
            self.sigma_v_X, self.sigma_tau_X = self.sigma_X[0], self.sigma_X[1]

            if(len(self.params_Z)==2):
                self.v_Z, self.tau_Z = abs(self.params_Z[0]), self.params_Z[1]
            
                self.sigma_v_Z, self.sigma_tau_Z = self.sigma_Z[0], self.sigma_Z[1]

            else:
                self.v_Z, self.tau_Z = abs(self.params_Z[0]), np.nan
            
                self.sigma_v_Z, self.sigma_tau_Z = self.sigma_Z[0], np.nan

#            # calculate the local slope of MSD and its asymptotes
#            self.Delays_Z, self.Slope_Z = self.computeLocalSlope(self.delays, self.MSD_Z)
#            self.Delays_X, self.Slope_X = self.computeLocalSlope(self.delays, self.MSD_X)

#             Number of time points corresponding to 1s of data
#            ind_5s = int(5/(self.Delays_Z[1] - self.Delays_Z[0]))
#
#            ind_1s = int(1/(self.Delays_Z[1] - self.Delays_Z[0]))
#
#            print('1 s data points: {}'.format(ind_1s))
#            print('5 s data points: {}'.format(ind_5s))
#                
#            index_x = next((i for i,x in enumerate(self.Delays_X) if x >= 10*self.tau_X), None)
#
#            if(index_x is None):
#                index_x = len(self.Delays_X)
#
#            print(index_x)
#
#            self.slope_asymp_X = np.nanmean(self.Slope_X[index_x - ind_1s : index_x])
#
#
#            self.slope_asymp_X_sigma = np.nanstd(self.Slope_X[index_x - ind_1s : index_x])
    

#            if(not np.isnan(self.tau_Z).any()):
#                index_z = next((i for i,x in enumerate(self.Delays_Z) if x >= 10*self.tau_Z), None)
#                if(index_z is None):
#                    index_z = len(self.Delays_Z)
#                print(index_z)
#                self.slope_asymp_Z = np.nanmean(self.Slope_Z[index_z - ind_5s : index_z])
#                self.slope_asymp_Z_sigma = np.nanstd(self.Slope_Z[index_z - ind_5s : index_z])
#
#            else:
#                self.slope_asymp_Z = np.nanmean(self.Slope_Z[-ind_1s : -1])
#                self.slope_asymp_Z_sigma = np.nanstd(self.Slope_Z[-ind_1s : -1])
#
#            print('Slope asymptote Z: {} +- {}'.format(self.slope_asymp_Z, self.slope_asymp_Z_sigma))
#            print('Slope asymptote X: {} +- {}'.format(self.slope_asymp_X, self.slope_asymp_X_sigma))


            # Calculate the mean and standard deviation of the velocity time series

#            self.calculate_velocityDist()
            self.getMeanStd_velocity()

            
            self.slope_asymp_Z, self.slope_asymp_Z_sigma = self.computeLocalSlope_MSD_vs_variability(Time=Time, Traj_Z =Traj_Z, time_window = 5)


            # Save the resulting fit parameters in a file

            # params_df = pd.DataFrame({'Organism':[self.Organism], 'Condition':[self.Condition], 'OrgSize_mean':[self.OrgSize_mean], 'OrgSize_std':[self.OrgSize_std], 'v_X':[self.v_X], 'sigma_v_X': [self.sigma_v_X],'tau_X':[self.tau_X], 'sigma_tau_X':[self.sigma_tau_X], 'v_Z':[self.v_Z], 'sigma_v_Z':[self.sigma_v_Z], 'tau_Z':[self.tau_Z], 'sigma_tau_Z':[self.sigma_tau_Z], 'MSD_Slope_Z':[self.slope_asymp_Z], 'MSD_Slope_Z_sigma':[self.slope_asymp_Z_sigma], 'MSD_Slope_X':[self.slope_asymp_X], 'MSD_Slope_X_sigma':[self.slope_asymp_X_sigma], 'Vx_variability_mean':[self.Vx_variability_mean], 'Vx_variability_std':[self.Vx_variability_std], 'Vz_variability_mean':[self.Vz_variability_mean], 'Vz_variability_std':[self.Vz_variability_std]})
            params_df = pd.DataFrame({'Organism':[self.Organism], 'Condition':[self.Condition], 'OrgSize_mean':[self.OrgSize_mean], 'OrgSize_std':[self.OrgSize_std], 'v_X':[self.v_X], 'sigma_v_X': [self.sigma_v_X],'tau_X':[self.tau_X], 'sigma_tau_X':[self.sigma_tau_X], 'v_Z':[self.v_Z], 'sigma_v_Z':[self.sigma_v_Z], 'tau_Z':[self.tau_Z], 'sigma_tau_Z':[self.sigma_tau_Z], 'coeff_det_x':[self.coeff_det_X], 'coeff_det_z':[self.coeff_det_Z],  'MSD_Slope_Z':[self.slope_asymp_Z], 'MSD_Slope_Z_sigma':[self.slope_asymp_Z_sigma], 'Vx_mean':[self.Vx_mean], 'Vx_std':[self.Vx_std], 'Vz_mean':[self.Vz_mean], 'Vz_std':[self.Vz_std]})
#            params_df = pd.DataFrame({'Organism':[self.Organism], 'Condition':[self.Condition], 'OrgSize_mean':[self.OrgSize_mean], 'OrgSize_std':[self.OrgSize_std], 'v_X':[self.v_X], 'sigma_v_X': [self.sigma_v_X],'tau_X':[self.tau_X], 'sigma_tau_X':[self.sigma_tau_X], 'v_Z':[self.v_Z], 'sigma_v_Z':[self.sigma_v_Z], 'tau_Z':[self.tau_Z], 'sigma_tau_Z':[self.sigma_tau_Z], 'coeff_det_x':[self.coeff_det_X], 'coeff_det_z':[self.coeff_det_Z]})

            params_df.to_csv(os.path.join(self.fittingFolder, saveFile))

        else:

            print('Loading fitted parameters from file ...')
            params_df = pd.read_csv(os.path.join(self.fittingFolder, saveFile))

            self.v_X, self.tau_X = params_df['v_X'], params_df['tau_X']
            
            self.sigma_v_X, self.sigma_tau_X = params_df['sigma_v_X'], params_df['sigma_tau_X']

            self.v_Z, self.tau_Z = params_df['v_Z'], params_df['tau_Z']
            
            self.sigma_v_Z, self.sigma_tau_Z = params_df['sigma_v_Z'], params_df['sigma_tau_Z']


            self.params_X = np.array([self.v_X, self.tau_X])

            if(not np.isnan(self.tau_Z).any()):
                self.params_Z = np.array([self.v_Z, self.tau_Z])
            else:
                self.params_Z = np.array([self.v_Z])

            print(self.params_X)
            print(self.params_Z)


    def computeLocalSlope_MSD_vs_variability(self, Time = None, Traj_Z = None, time_window = 10):
        
        # Calculating Slope using WLS ICE fit
        print('Calculating Slope using WLS ICE fit...')
        
        starttime = np.max(self.delays) - time_window            # first time point to include in fitting
        stoptime = np.max(self.delays)
        epsilon0 = Time[1] - Time[0] # time step size
        start_idx = int(starttime / epsilon0)
        stop_idx = int(stoptime / epsilon0)
        
      
        sub_Traj_Z = Traj_Z[:,start_idx:stop_idx]
        sub_Time = Time[start_idx:stop_idx]
        M,N = np.shape(sub_Traj_Z)
        assert(N == len(sub_Time))
        
        # Perform the fit
        min_method = "nm"           # Use Nelder-Mead minimization method
        guess=np.array([.5,.5])     # Starting point in parameter space
        # The analytical function to fit, its gradient, and hessian
        
        wlsice.init(self.f_powerlaw, self.df_powerlaw, self.d2f_powerlaw)
        
        # Perform the actual fit
        params, sigma, chi2_min = wlsice.fit(sub_Time, sub_Traj_Z, guess, min_method)
        
        # RESULT:
        print("# trajectories M=%s,\tsampling times N=%s, t_0=%s" % (M, N, sub_Time[0]))
        print("# Optimal param:\t%s" % params)
        print("# Sigma:\t%s" % sigma)
        print("# Chi-square value: \t%s" % chi2_min)
        
        # % error in slope estimate
        print('% error in slope estimate: {}'.format(100*sigma[1]/params[1]))
        
        localSlope_MSD_Z = params[1]
        stdError_localSlope_MSD_Z = sigma[1]
        
        # Plot the MSD of the track segment along with the fit:
        import PlotFunctions.PlotUtils as PlotUtils

        plt.figure()
        PlotUtils.errorfill(self.delays[start_idx:stop_idx], self.MSD_Z[start_idx:stop_idx], self.stdev_Z[start_idx:stop_idx], ax = plt.gca(), color = 'r', label = 'Vertical (Z)')
        plt.plot(sub_Time, self.f_powerlaw(sub_Time, params), 'k--')
        
        return localSlope_MSD_Z, stdError_localSlope_MSD_Z
#------------------------------------------------------------------------------
# Functions for velocity distrtibution calculation 
#------------------------------------------------------------------------------
                
    def calculate_velocityDist(self):
        
        
        saveFile = self.saveFile + '_' + '_VelocityTimeSeries'

        
        overwrite = False

        if(not os.path.exists(os.path.join(self.velocityDistFolder, saveFile)) or overwrite is True):
            Velocities_X = np.array([])
            Velocities_Y = np.array([])
            Velocities_Z = np.array([])

            # OrgDim_mean, OrgDim_std = self.get_Mean_Std('OrgDim')
            dataFrame_full = pd.DataFrame({'Organism':[],'Condition':[], 'Track':[], 'VelocityX':[],'VelocityY':[],'VelocityZ':[]})

            Vx_mean = np.zeros(self.nTracks)
            Vz_mean = np.zeros(self.nTracks)

            Vx_std = np.zeros(self.nTracks)
            Vz_std = np.zeros(self.nTracks)

            for ii in range(self.nTracks):

                

                
                currTrack = self.Tracks[ii]
                
                print('...Analyzing Track {} of Duration {} s'.format(ii, np.max(currTrack.T)))

#                AutoTracking = np.array(~currTrack.df['Manual Tracking'], dtype= 'bool')
#
#                Velocities_X = np.array(currTrack.Vx[AutoTracking])
#                Velocities_Y = np.array(currTrack.Vy[AutoTracking])
#                Velocities_Z = np.array(currTrack.Vz[AutoTracking])
                
            
#
#                Velocities_X = np.array(currTrack.Vx)
#                Velocities_Y = np.array(currTrack.Vy)
#                Velocities_Z = np.array(currTrack.Vz)
                
    
                
                Velocities_X = np.array(currTrack.Vx_smooth)
                Velocities_Y = np.array(currTrack.Vy_smooth)
                Velocities_Z = np.array(currTrack.Vz_smooth)
                
                dataLen = len(Velocities_X)

                Vx_mean[ii] = np.nanmean(Velocities_X)
                Vx_std[ii] = np.nanstd(Velocities_X)

                Vz_mean[ii] = np.nanmean(Velocities_Z)
                Vz_std[ii] = np.nanstd(Velocities_Z)


                dataFrame_full = dataFrame_full.append(pd.DataFrame({'Organism':np.repeat(self.Organism,dataLen,axis = 0),'Condition':np.repeat(self.Condition,dataLen,axis=0), 'Track': np.repeat(currTrack.path ,dataLen,axis=0) , 'VelocityX':Velocities_X,'VelocityY':Velocities_Y,'VelocityZ':Velocities_Z}))

                dataFrame_full.to_csv(os.path.join(self.velocityDistFolder, saveFile + currTrack.trackName + '.csv'))

            self.Vz_variability_mean = np.nanmean(Vz_std/Vz_mean)

            self.Vz_variability_std = np.nanstd(Vz_std/Vz_mean)

            self.Vx_variability_mean = np.nanmean(Vx_std/Vx_mean)

            self.Vx_variability_std = np.nanstd(Vx_std/Vx_mean)
            
            print(Vz_mean)
            print(Vz_std)
            
            print(Vx_mean)
            print(Vx_std)
            print('Mean and Std of variability across different tracks: {} +- {}'.format(self.Vz_variability_mean, self.Vz_variability_std))


        else:

            # Load the velocity time series from memory
            dataFrame_full = pd.read_csv(os.path.join(self.velocityDistFolder, saveFile))

      
        self.velocities_df = dataFrame_full


    def getMeanStd_velocity(self):

        if(self.velocities_df is not None):

            self.Vx_mean, self.Vx_std = np.nanmean(self.velocities_df['VelocityX']), np.nanstd(self.velocities_df['VelocityX'])

            self.Vz_mean, self.Vz_std = np.nanmean(self.velocities_df['VelocityZ']), np.nanstd(self.velocities_df['VelocityZ'])

            print('Velocity Z: {} +- {}'.format(self.Vz_mean, self.Vz_std))
            print('Velocity X: {} +- {}'.format(self.Vx_mean, self.Vx_std))
            

            
    def polarHistogram(self):
        
        # First calculate the velocity time-series or load from memory
        if(self.velocities_df is None):
            self.calculate_velocityDist()
            
        Vx = self.velocities_df['VelocityX']
        Vy = self.velocities_df['VelocityY']
        Vz = self.velocities_df['VelocityZ']
        
        vector_magnitude = (Vx**2 + Vy**2 + Vz**2)**(1/2)
        
        Orientation_vectors = np.zeros((3, len(Vx)))
        
        Orientation_vectors[0,:] = Vx/vector_magnitude
        Orientation_vectors[1,:] = Vy/vector_magnitude
        Orientation_vectors[2,:] = Vz/vector_magnitude
        
        # Extract the orientation angle from the vertical
        
        Z_gravity = [0, 0, 1]
        
        cos_theta = Orientation_vectors[0,:]*Z_gravity[0] + Orientation_vectors[1,:]*Z_gravity[1] + Orientation_vectors[2,:]*Z_gravity[2]

        # Theta value in degrees
        theta_array = np.arccos(cos_theta)*(180/(np.pi))
        
        
        # Plot a polar histogram of the angles
        
        bin_size = 10
        
        a , b = np.histogram(theta_array, bins=np.arange(0, 180 + bin_size, bin_size), density = True)
        
        centers = np.deg2rad(np.ediff1d(b)//2 + b[:-1])
        
        
        plt.figure()
        plt.hist(theta_array, density=True, color = 'b')
        plt.xlabel('Theta in degrees')
        plt.show()
        
        
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111, projection='polar')
        ax.bar(centers, a, width=np.deg2rad(bin_size), bottom=0.0, color='b', edgecolor='k')
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        plt.show()
        
        
        
        
        
        
        

#------------------------------------------------------------------------------
        # Entropy calculation functions
#------------------------------------------------------------------------------
    def init_entropy_params(self):
        # Initialize entropy parameters
        
        # Maximum scale factor for MSE analysis in seconds
        self.maxScaleFactor_time = 120
        
        # We calculate the entropy of the baseline white-noise signal using scaled number of data points to get better statistics.
        self.scale_factor_test_signal = 5
        
        self.getCommonScalefactors()
        
    def getCommonScalefactors(self):
        
        print('Max time difference : {}'.format(self.maxDiff))
        maxScale = int(np.floor(self.maxScaleFactor_time/self.maxDiff))
        
        print('Max scale : {}'.format(maxScale))
        
        self.ScaleFactors_common = np.array(range(10,maxScale,10), dtype = 'int') + 1
        
        self.ScaleFactors_common_time = self.maxDiff*self.ScaleFactors_common
        
            
    def get_mse_tracks(self, overwrite = False):
        
        print('Calculating MSE for tracks ...')
        
        saveFile = self.Organism + '_' + str(self.Condition) + '_' + 'MSE.csv'
        
        savePath = os.path.join(self.entropyFolder, saveFile)
        
        if(not os.path.exists(savePath) or overwrite == True):
        
            # Save the MSEs into a CSV file
            
            df_mse = pd.DataFrame({'Organism':[], 'Condition':[], 'Track name':[],'Scale factor times':[], 'MSE_Vz': [], 'MSE_noise': []})
            
            
            axes = None
       
            # Compute the MSE for a set of tracks from the same condition
            
            for ii in range(self.nTracks):
                
                
                currTrack = self.Tracks[ii]
            
                print('...Analyzing Track {} of Duration {} s'.format(ii, np.max(currTrack.T)))
                
                Time = currTrack.T
                
                Signal = currTrack.Vz
                
                print('No:of track data points: {}'.format(len(Time)))
                
                # We want to calculate the r value after excluding outliers
                tol = 0.2*np.nanstd(currTrack.Vz_smooth)
                
                std = np.nanstd(currTrack.Vz_smooth)
                mean = np.nanmean(currTrack.Vz_smooth)
                
                test_signal_time = np.linspace(0, np.max(Time), self.scale_factor_test_signal*len(Time))
                test_signal = mean + np.random.normal(0, std, self.scale_factor_test_signal*len(Time))
                
                maxScale = int(np.ceil(self.maxScaleFactor_time/currTrack.dT))
                
                ScaleFactors = np.array(range(maxScale), dtype = 'int') + 1
    
                ScaleFactors_time = currTrack.dT*ScaleFactors
                
                
                # calculate the MSE
                mse = ent.multiscale_entropy(Signal, 3, tol, maxScale)
                
                mse_test = ent.multiscale_entropy(test_signal, 3, tol, maxScale)
                
                mse_fun = interpolate.interp1d(ScaleFactors_time, mse, 'linear')
                
                mse_fun_test = interpolate.interp1d(ScaleFactors_time, mse_test, 'linear')
                
                
                # Get the interpolated entropies at common times
                mse_common = mse_fun(self.ScaleFactors_common_time)
                
                mse_common_test = mse_fun_test(self.ScaleFactors_common_time)
                
                
                # Plot the Signal and MSEs
                axes = self.plot_signal_entropy(Time, Signal, self.ScaleFactors_common_time, mse_common, axes)
                self.plot_signal_entropy(test_signal_time, test_signal, self.ScaleFactors_common_time, mse_common_test, axes, label = 'Noise test')
                
              
                
                data_len = len(mse_common)
                
                df_mse = df_mse.append(pd.DataFrame({'Organism':np.repeat(self.Organism, data_len, axis = 0), 'Condition':np.repeat(self.Condition, data_len, axis = 0), 'Track name':np.repeat(currTrack.trackName, data_len, axis = 0), 'Scale factor times': self.ScaleFactors_common_time, 'MSE_Vz': mse_common, 'MSE_noise':mse_common_test}))
        
        
            # Save the dataFrame as a CSV file
            
            df_mse.to_csv(savePath)
            
            plt.savefig(self.Organism + '_' + str(self.Condition) + '_MSE.png', dpi = 150)
            plt.savefig(self.Organism + '_' + str(self.Condition) + '_MSE.svg', dpi = 150)


#------------------------------------------------------------------------------
        # Plotting functions
#------------------------------------------------------------------------------
    def plot_signal_entropy(self, time, signal, scale_factors, mse, axes, label = None):
        
        if(label is None):
            label = self.Organism + '_' + str(self.Condition)
    
        if(axes is None):
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,12))
            
        axes[0].plot(time, signal, label = label)
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Signal')
    
        axes[1].scatter(scale_factors, mse, 40, label = label)
        axes[1].plot(scale_factors, mse, label = label)
        axes[1].set_xlabel('Scale factor')
        axes[1].set_ylabel('MSE')
        
        plt.title(self.Organism + '_' + str(self.Condition))
        plt.legend()
       
        
        return axes
        
        
        
        
    def plot_velocityDist(self):


        if(self.velocities_df is not None):

            dataFrame_full = self.velocities_df


            my_pal = {'VelocityZ_noWall': 'b' ,'VelocityX_noWall': 'r'}
            color_list = sns.color_palette("RdBu_r", 7)
            #my_pal = {'VelocityZ_noWall': color_list[0] ,'VelocityX_noWall': color_list[6]}

            xlim1 = -2
            xlim2 = 2
            decimals = 1

            plt.figure(figsize=(4.5,4))
            ax0 = sns.distplot(dataFrame_full.loc[dataFrame_full["Organism"] == self.Organism,"VelocityZ"],  kde = True , color = 'b', norm_hist = True, hist_kws={"histtype": "bar","edgecolor":'w', "linewidth": 0.2, "alpha": 0.5, "label":'Vz'})
            #ax0 = sns.distplot(dataFrame_full.loc[dataFrame_full["Organism"] == Organism2,"VelocityZ"],  kde = True , color = 'k', norm_hist = True, hist_kws={"histtype": "bar","edgecolor":'w', "linewidth": 0.2, "alpha": 0.5, "label":'Vz'})

            ax1 = sns.distplot(dataFrame_full.loc[dataFrame_full["Organism"] == self.Organism,"VelocityX"],  kde = True , color = 'r', norm_hist = True, hist_kws={"histtype": "bar","edgecolor":'w', "linewidth": 0.2, "alpha": 0.5, "label":'Vx'})  

            plt.xlim([xlim1, xlim2])
            plt.xticks(np.round(np.linspace(xlim1,xlim2,5), decimals=decimals))
            #plt.savefig(os.path.join(saveSubFolder,Organism+'VelocityDistribution_FINAL.svg'))
            plt.xlabel('Velocity')
            plt.ylabel('PDF')
            plt.legend()
            ax = plt.gca()
            ax.set_aspect(0.6)
            plt.show()

            my_pal = {'VelocityZ': 'b' ,'VelocityX': 'r'}

            plt.savefig(self.Organism+'_'+str(self.Condition)+'.svg', dpi = 150)
        
                
    def plotMSD(self, figname = 1, plot_analytical = False, plot_fit = False, savefig = False):
        

#        f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)


        
        plt.figure(figsize = (8,6))
        ax0 = plt.gca()
        PlotUtils.errorfill(self.delays, self.MSD_X, self.stdev_X, ax = ax0, color = 'r', label = 'Horizontal (X)')
        
        PlotUtils.errorfill(self.delays, self.MSD_Z, self.stdev_Z, ax = ax0, color = 'b', label = 'Vertical (Z)')
#        
        if(plot_analytical==True and self.testFlag==True):
            ax0.plot(self.delays, self.DiffCoeff*self.delays, color = 'k', marker = 'o')


        ax0.set_xlabel('Time (s)')
        ax0.set_ylabel('MSD')
        plt.xlim(0,np.max(self.delays))
        plt.legend(loc='upper left', frameon=False)
        
        
        if(plot_fit):
            if(np.isnan(self.tau_Z).any()):
                ax0.plot(self.delays, self.f_ballistic(self.delays, self.params_Z), 'k-', label = 'Z - WLS-ICE Fit (Ballistic')
            else:
                ax0.plot(self.delays, self.f(self.delays, self.params_Z), 'k-', label = 'Z - WLS-ICE Fit (Taylor equation)')

        plt.title(self.Organism + '_' + str(self.Condition))

        if(savefig is True):
            plt.savefig(os.path.join(self.figuresFolder,self.saveFile+'_MSD_Linear.svg'),bbox_inches='tight',dpi=150)
            plt.savefig(os.path.join(self.figuresFolder,self.saveFile+'_MSD_Linear.png'),bbox_inches='tight',dpi=150)
        plt.show()
        
        
        # 
        time = self.delays
        A = 0.1
        B = 1
        X_linear = B*time
        X_quad = A*(time**2)
        
        
        plt.figure(figsize = (8,6))
        ax1 = plt.gca()
        ax1.scatter(self.delays, self.MSD_X, 20, color = 'r', marker = 'o', label = 'Horizontal (X)', alpha = 0.5)

        if(plot_fit is True):

            # X displacement fit with Taylor function
            ax1.plot(self.delays, self.f(self.delays, self.params_X), 'k--', label = 'X - WLS-ICE Fit (Taylor equation)')
        
        ax1.scatter(self.delays, self.MSD_Z, 20, color = 'b', marker = 's', label = 'Vertical (Z)', alpha =0.5)#        ax3.set_yscale('log')

        if(plot_fit is True):
            # X displacement fit with Taylor function
            if(np.isnan(self.tau_Z).any()):
                ax1.plot(self.delays, self.f_ballistic(self.delays, self.params_Z), 'k-', label = 'Z - WLS-ICE Fit (Ballistic')
            else:
                ax1.plot(self.delays, self.f(self.delays, self.params_Z), 'k-', label = 'Z - WLS-ICE Fit (Taylor equation)')


        ax1.plot(time,X_linear,'g--', linewidth=1)
        ax1.plot(time,X_quad, 'g-', linewidth=1)
        ax1.set_yscale('log')
        ax1.set_xscale('log')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('MSD')
        
        # plt.xlim(0,np.max(self.delays))
        plt.legend(loc='upper left', frameon=False)
    

        plt.title(self.Organism + '_' + str(self.Condition))

        if(savefig is True):
            plt.savefig(os.path.join(self.figuresFolder,self.saveFile+'_MSD_Log.svg'),bbox_inches='tight',dpi=150)
            plt.savefig(os.path.join(self.figuresFolder,self.saveFile+'_MSD_Log.png'),bbox_inches='tight',dpi=150)
        plt.show()


    def plotLocalSlope(self, savefig = False):

        plt.figure()
        plt.plot(self.Delays_X, self.Slope_X, color = 'r', marker = 'o', linestyle = '--', label = 'Horizontal (X)')
        plt.plot(self.Delays_Z, self.Slope_Z, color = 'b', marker = 's', linestyle = '-', label = 'Vertical (Z)')
        plt.ylim(0,2.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Local slope of MSD curve')
        plt.title(self.Organism + '_' + str(self.Condition))

        plt.legend(loc='best', frameon=False)

        if(savefig is True):
            plt.savefig(os.path.join(self.figuresFolder,self.saveFile+'_MSD_LocalSlope.svg'),bbox_inches='tight',dpi=150)
            plt.savefig(os.path.join(self.figuresFolder,self.saveFile+'_MSD_LocalSlope.png'),bbox_inches='tight',dpi=150)

        plt.show()









        
        
        
        
  
        
        
        

# def main():
    
  

    
    
    
    
    
    
# if __name__ == '__main__':
#     main()    
        

        
        
