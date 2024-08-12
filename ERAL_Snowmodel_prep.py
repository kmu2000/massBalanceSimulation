# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 10:06:02 2023

@author: Kriti.Mukherjee

This script extracts the hourly ERA5 Land data for each month and writes the required variables
for SnowModel in CSV files. It then merges all of them together and converts the cumulative 
precipitation values to hourly total precipitation and prepares a final csv met file.
"""

import glob
import os
import numpy as np
import pandas as pd
from calendar import monthrange
from netCDF4 import Dataset as dt
import warnings
warnings.filterwarnings('ignore')


def time_df(sy, ey, sm, em):
    # create the data frame for the time steps
    dfDate = pd.DataFrame()
    for i in range(sy,ey): # enter the years for your data       
        year = []
        month = []
        day = []
        hour = []
        months = np.arange(1,13)
        hours = np.arange(24)
        for m in months:
            r = monthrange(i,m)
            days = np.arange(1,r[1]+1)
            for d in days:
                for h in hours:
                    year.append(i)
                    month.append(m)
                    day.append(d)
                    hour.append(h)

        df = pd.DataFrame({'Year':year, 'Month':month, 'Day':day, 'Hour':hour})     
        dfDate = dfDate.append(df, ignore_index=True)
    
    return dfDate

def metdataprep(ncdata):
    # Latent heat of vaporization in J/Kg
    L = 2.453*10**6
    
    # Gas constant for moist air in J/Kg
    Rv = 461
    
    C = L/Rv
    
    # read the parameters 
    T2 = np.array(ncdata.variables['t2m'][:, :, :], dtype=np.float32)
    t2m = np.mean(T2,axis=(1,2))
    
    D2 = np.array(ncdata.variables['d2m'][:, :, :], dtype=np.float32)
    d2m = np.mean(D2,axis=(1,2))   
    
    P = np.array(ncdata.variables['tp'][:, :, :], dtype=np.float32)
    tp = np.mean(P,axis=(1,2))*1000
    
    V = np.array(ncdata.variables['v10'][:, :, :], dtype=np.float32)
    v10 = np.mean(V,axis=(1,2))
    
    U = np.array(ncdata.variables['u10'][:, :, :], dtype=np.float32)
    u10 = np.mean(U,axis=(1,2))
    
    # Wind speed 
    W = np.sqrt(u10 * u10 + v10 * v10)
    winDir = 180 + np.arctan2(u10, v10) * 180 / np.pi
    
    # reference temperature
    Tref = np.full(T2.shape[0],273.15)
    
    # relative humidity 
    rhn = (1/Tref - 1/t2m)
    rhd = (1/Tref - 1/d2m)
    es = 6.11*np.exp(C*rhn)
    e = 6.11*np.exp(C*rhd)
    rh = e/es*100
    
    
    df = pd.DataFrame({'T2':t2m, 'RH2':rh, 'U2':W, 'WDir':winDir, 'RRR':tp})
    return df

path = '/mnt/e/Glaciology/climate/ERA5/RGI15-02999/hourly/'

os.chdir(path)
data = glob.glob('*.zip')

if not os.path.exists('tmp'):
    os.mkdir('tmp')


for i in range(len(data)):
    #print('unzip -p ' + data[i] + ' data.nc > tmp/' + data[i][:-10] + 'nc')
    os.system('unzip -p ' + data[i] + ' data.nc > tmp/' + data[i][:-10] + 'nc')
    

# prepare met data in csv format
os.chdir(path + 'tmp')
metdata = glob.glob('*.nc')

for i in range(len(metdata)):
    print('prepare data for: ', i)
    year = int(metdata[i][5:9])
    month = int(metdata[i][10:12])
    
    ncdata = dt(metdata[i],'r')
    
    #craete timestamp
    Time = []  
    r = monthrange(year,month)
    days = np.arange(1,r[1]+1)
    hours = np.arange(24)
    Y = []
    M = []
    Day = []
    Hour = []
    for k in days:
        for j in hours:
            timePeriod = pd.Period(freq ='H', year = year, month = month, day = k,
                                 hour = j)
            # converting period to timestamp
            timeStamp =timePeriod.to_timestamp()
            formatted_ts = timeStamp.strftime('%Y %m %d %H')
            # print(formatted_ts)
            Y.append(formatted_ts[:4])
            M.append(formatted_ts[5:7])
            Day.append(formatted_ts[8:10])
            Hour.append(formatted_ts[11:13])
            
    
    if year == 1950 and month == 1:
        Y = Y[1:]
        M = M[1:]
        Day = Day[1:]
        Hour = Hour[1:] 
    
    dfDate = pd.DataFrame({'Year':Y,'Month':M, 'Day':Day,'Hour':Hour})
    dfmet = metdataprep(ncdata)
    df = pd.concat([dfDate, dfmet], axis=1)
    df.to_csv(path + 'tmp/' + metdata[i][:-3] + '.csv', index=False )
    

to_merge = sorted(glob.glob('*.csv'))
print(to_merge)
dfs = []
for filename in to_merge:
    # read the csv
    df = pd.read_csv(filename, header=None, skiprows=1)
    dfs.append(df)

# concatenate them horizontally
merged = pd.concat(dfs)
merged = merged.iloc[1:]
merged.columns = ['Year','Month','Day','Hour','T2','RH2','U2','WDIR','RRR']

print(merged.shape)
# convert from cumulative to original hourly values
Ppt = merged['RRR'].values


lenData = int(len(Ppt)/24)
print(lenData)
Rain = []

dfm = []
for i in range(lenData+1):
    col1 = Ppt[(i * 24):(24 * (i + 1))]
    
    df1 = pd.DataFrame({'ppt':col1})
    df2 = df1.diff().fillna(df1.iloc[0])
        
    dfm.append(df2)
    

mergeMet = pd.concat(dfm)

print(mergeMet)

#drop the original columns and replace by new ones
dfmet = merged.drop(['RRR'], axis=1)
#dfmet = dfmet.iloc[:-23]
#print(dfmet.shape)
dfmet['RRR'] = mergeMet['ppt'].values
dfmet['T'] = dfmet['T2']-273.15
dfmet = dfmet.drop(['T2'], axis=1)
dfmet['stn'] = '101'
dfmet['X'] = 667828.04
dfmet['Y'] = 4766205.48
dfmet['Z'] = 3750

dfmet = dfmet[['Year','Month','Day','Hour','stn','X','Y','Z','T','RH2','U2','WDIR','RRR']]

metoutpath ='/mnt/e/lidar3/SnowModel/RGI15-02999/climate/'
if not os.path.exists(metoutpath):
    os.makedirs(metoutpath)
dfmet.to_csv(metoutpath + 'metERA.csv', index=False)

os.system('rm -rf ' + path + 'tmp/' )
