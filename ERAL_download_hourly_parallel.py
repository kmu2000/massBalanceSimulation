# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 06:23:47 2024

@author: kriti.mukherjee
code to download ERA5 Land hourly data
"""

import cdsapi
import os
import multiprocessing
import numpy as np


c = cdsapi.Client()
def download(year, month, variable, area):    
    c.retrieve(
        'reanalysis-era5-land',
        {
            'variable': variable,
            'year': year,
            'month': month,
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',
            ],
            'time': [
                '00:00', '01:00', '02:00',
                '03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00',
                '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00',
                '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00',
                '21:00', '22:00', '23:00',
            ],
            'area': area,
            'format': 'netcdf.zip',
        },
        Path + 'ERAL_' + year + '_' + month + '.netcdf.zip')

loc = 'RGI15-02999'
Path = '/mnt/e/Glaciology/climate/ERA5/' + loc + '/hourly/'


area = [27.8, 88.30, 27.7,88.40] 
variables = [
            '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_dewpoint_temperature',
            '2m_temperature', 'total_precipitation',
        ]
months = ['01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',]

years = [str(i) for i in np.arange(1969, 1970)]   
print(years)


if __name__== '__main__':
    processes = []
    for i in years:
        for j in months:
            p = multiprocessing.Process(target=download,args=(i,j,variables,area))
            processes.append(p)
            p.start()

    for process in processes:
        process.join()

print("All done!")