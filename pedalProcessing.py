# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 15:28:25 2016

@author: vijay
@motivation: tzhou
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
path = "../tripCsvFiles/"
PATH = "../pickleFiles/"

#this function is to remove small hills in brake data
def reprocessZeroCross(brakePressure, zeroCros, thresh, window):  
    zeroCrosRefined = []
    for i in range(len(zeroCros)):
        if  i < len(zeroCros)-1:
            if max(brakePressure[zeroCros[i]:zeroCros[i+1]]) > thresh:
                zeroCrosRefined.append(zeroCros[i])
    if max(brakePressure[zeroCros[-1]:len(brakePressure)-1]) > thresh:
        zeroCrosRefined.append(zeroCros[-1])
    return zeroCrosRefined
    

def processPedalData(brakePressure, accelPressure, slopeTr=20, thresh =0.1, show=0,fig=1): #give brakePressure as list or numpy array
    brakePressure = np.array(brakePressure)
    accelPressure = np.array(accelPressure)
    N = len(brakePressure)
    # get zero crossing
    shiftedbrakePressure = brakePressure - 0.001
    shiftedaccelPressure = accelPressure - 0.001
    zeroCros = []
    for i in range(N-1):
        if shiftedbrakePressure[i+1] >= 0 and shiftedbrakePressure[i] < 0:
            zeroCros.append(i)
    bzeroCrosRefined = reprocessZeroCross(brakePressure, zeroCros, thresh, window = slopeTr) #20 implies 2 sec window.
    
    zeroCros = []
    for i in range(N-1):
        if shiftedaccelPressure[i+1] >= 0 and shiftedaccelPressure[i] < 0:
            zeroCros.append(i)
    azeroCrosRefined = reprocessZeroCross(accelPressure, zeroCros, thresh, window = slopeTr) #20 implies 2 sec window.
    
    print "Number of braking events: %i" % len(bzeroCrosRefined)
    print "Number of accel events: %i" % len(azeroCrosRefined)
        
    
    
    # plot them
    if show:
        #plt.subplot(10,1,1)
        """
        plt.figure(fig)
        plt.plot(brakePressureSlope, 'g-')
        plt.title('brake pressure slope')
        
        plt.subplot(10,1,2)
        plt.plot(brakePressure, 'g-')
        plt.plot(uprise, brakePressure[uprise], 'r*')
        plt.title('uprise edge')

        plt.subplot(10,1,3)
        plt.plot(brakePressure, 'g-')
        plt.plot(downrise, brakePressure[downrise], 'r*')
        plt.title('downrise edge')

        plt.subplot(10,1,4)
        plt.plot(brakePressure, 'g-')
        plt.plot(nonzero, brakePressure[nonzero], 'r*')
        plt.title('non zero value')
        
        plt.subplot(10,1,5)
        plt.plot(brakePressureSlope, 'g-')
        plt.plot(localMax, brakePressureSlope[localMax], 'r*')
        plt.title('brake pressure slope local max')
        
        #plt.subplot(10,1,6)
        plt.figure(fig+1)        
        plt.plot(brakePressure, 'g-')
        plt.plot(localMax, brakePressure[localMax], 'r*', markersize = 10)
        plt.title('brake pressure local max')
        """
        plt.figure(fig+2)        
        plt.plot(brakePressure, 'g-')
        plt.plot(bzeroCrosRefined, np.ones(N)[bzeroCrosRefined], 'b*', markersize=10)
        plt.title('brake pressure')
        
        plt.figure(fig+3)        
        plt.plot(accelPressure, 'g-')
        plt.plot(azeroCrosRefined, 20*np.ones(N)[azeroCrosRefined], 'b*', markersize=10)
        plt.title('accel pressure')
        
        """
        #plt.subplot(10,1,7)
        plt.figure(fig+4)        
        plt.plot(brakePressure, 'g-')
        plt.plot(zeroCrosRefined, brakePressure[zeroCrosRefined], 'r*', markersize = 10)
        plt.title('brake pressure zero crossing')
        """        
        

        plt.show()

    # format label
    
    ybrake = [1 if i in bzeroCrosRefined else 0 for i in range(N) ]
    yaccel = [1 if i in azeroCrosRefined else 0 for i in range(N) ]
    
    
    
    return np.array(ybrake), np.array(yaccel)
    
def resample(df):
    RESAMPLE = '100L'
    #df = df[df.index!=df.index[0]]
    #df = df[df.index!=df.index[-1]]
    df = df.resample(RESAMPLE).first()
    #df = df.interpolate()
    
    return df['pbrk'].values.tolist(),df

def load_data(file): 
    with open(file, 'rb') as f:
        return pickle.load(f)    
    
if __name__ == '__main__':
    file = '1463696819302037'
    """end_csv = '_features_agg.csv'
    destfile = pd.read_csv(path+file+end_csv, skiprows=(1,2))
    destfile = destfile.set_index(destfile['timestamp'])
    brakePressureSampled = destfile['pbrk'].values.tolist()
    accelPressureSampled = destfile['hv_accp'].values.tolist()"""
    
    end_pk = '_extr.pkl'
    #index = {'brake':1, 'speed':3, 'accel':2, 'str_angle':0, 'lane':13}
    indices = [1,2]
    data = load_data(PATH+file+end_pk)
    frames = [items['data'].groupby(level=0).first() for ind, items in enumerate(data) if ind in indices]
    del data
    index = frames[-1].index
    frames_sync = [frame.reindex(index=index, method='ffill') for frame in frames]
    destfile = pd.concat(frames_sync, axis=1)
    destfile = destfile.resample('100L').ffill()
    destfile.fillna(0,inplace=True)
    brakePressureSampled = destfile['pbrk'].values.tolist()
    accelPressureSampled = destfile['hv_accp'].values.tolist()
    #brakecsv = brakecsv.set_index(pd.DatetimeIndex(brakecsv['timestamp']))
    #brakecsv.index = pd.to_datetime(brakecsv.index, unit='us')
    #brakePressureSampled, brakecsv = resample(brakecsv)
    
    #brakePressureSampled = brakecsv.resample('100L').first()
    #brakePressure = brakecsv['pbrk'].values.tolist()
    
    #ySlopeLocalMax, yzeroCrossing = processBrakeData(brakePressure, slopeTr = 0.05, show = 1, fig =1)
    ybrake, yaccel = processPedalData(brakePressureSampled,accelPressureSampled, slopeTr = 20, thresh=0.1, show = 1,fig =4)
    #brakecsv['zeroCross'] = ySparseZeroCrossingSampled    
    #print brakecsv

    
 
