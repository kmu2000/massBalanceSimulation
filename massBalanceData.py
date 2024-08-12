# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 08:11:12 2023

@author: Kriti.Mukherjee
"""

import os
from MB_CalcFunc import prepdf, mbmean, mbpoly, mb_uncertainty
import glob
import pandas as pd


pathdem = '/mnt/e/Glaciology/MassBalance/coreg_all/'
dem_s = '20140926_PHR_CS_CoregDEM_30m.tif'

yearS = dem_s[:4]
yearE = '2020'
period = int(yearE)-int(yearS)
dem_e = glob.glob(pathdem + yearE + '*.tif')

chhotadem = pathdem + 'chhota/'
if not os.path.exists(chhotadem):
    os.makedirs(chhotadem)
gs = 'chhota_' + dem_s
#ge = 'chhota_' + dem_e

pathdh = pathdem + 'DH/'

dh = []
ge = []
dhy = []
for i in range(len(dem_e)):
    ge.append('chhota_' + dem_e[i][40:48] + '.tif')
    dh.append('dh_' + dem_e[i][40:48] + '_' + dem_s[:8] + '.tif')
    dhy.append('dh_' + dem_e[i][40:48] + '_' + dem_s[:8] + '_annual.tif')
   


path_glapoly = '/mnt/e/lidar3/SnowModel/ChhotaShigri/Extents/'
poly = 'ChhotaShigri_2020.shp'
glapoly = path_glapoly + poly

pathMask = '/mnt/e/Glaciology/MassBalance/GlacierMask/'
mask = '2011_GlaciervegMask_Chhota.tif'
outMask = 'final_' + mask

SRTM = '/mnt/e/Glaciology/DEM/SRTM/'
dem = 'n32_e077_1arc_v3_32643.tif'
demgla = 'Chhota_' + dem


### to extract the glacier area use the following lat/lon corners
longitude_upper_left = '77.475'
latitude_upper_left = '32.28'
longitude_lower_right = '77.56'
latitude_lower_right = '32.18' 

# extract glacier from the start dem
os.system('gdalwarp -overwrite -cutline ' + glapoly + ' ' + ' -te ' + longitude_upper_left 
          + ' ' + latitude_lower_right + ' ' + longitude_lower_right + ' ' + latitude_upper_left +  
          ' -te_srs EPSG:4326 -tr 10 10 ' + ' -r average -dstnodata -9999 ' + 
          pathdem + dem_s + ' ' + chhotadem + gs)


# extract glacier from the srtm dem to be used for elevation band data
os.system('gdalwarp -overwrite -cutline ' + glapoly + ' ' + ' -te ' + longitude_upper_left 
          + ' ' + latitude_lower_right + ' ' + longitude_lower_right + ' ' + latitude_upper_left +  
          ' -te_srs EPSG:4326 -tr 10 10 ' + ' -r average -dstnodata -9999 ' + 
          SRTM + dem + ' ' + pathdh + demgla)

# extract glacier from the end dems and dh 
for i in range(len(dem_e)):
    os.system('gdalwarp -overwrite -cutline ' + glapoly + ' ' + ' -te ' + longitude_upper_left 
          + ' ' + latitude_lower_right + ' ' + longitude_lower_right + ' ' + latitude_upper_left +  
          ' -te_srs EPSG:4326 -tr 10 10 ' + ' -r average -dstnodata -9999 ' 
          + dem_e[i] + ' ' + chhotadem + ge[i])

    os.system('gdal_calc.py --calc="(A-B)" --outfile=' + pathdh+dh[i] + ' -A ' + chhotadem + ge[i] 
          + ' -B ' + chhotadem + gs)
    os.system('gdal_calc.py --calc="(A-B)/{period}" --outfile={outfile} -A {A} -B {B}'.format(
        period=period,
        outfile=pathdh+dhy[i],
        A = chhotadem + ge[i],
        B = chhotadem + gs))


# extract glacier region from the glacier mask
os.system('gdalwarp -overwrite ' + ' -te ' + longitude_upper_left 
          + ' ' + latitude_lower_right + ' ' + longitude_lower_right + ' ' + latitude_upper_left +  
          ' -te_srs EPSG:4326 -tr 10 10 -r near ' 
          + pathMask + mask + ' ' + pathMask + outMask)



if __name__=='__main__':
    S = []
    E = []
    MB = [] 
    PL = []
    volM = []
    Area = []
    MBP = []
    volP = []
    SM = []
    SP = []
    # calculate mass balance using mean of each elevation band
    for i in range(len(dem_e)):
        df1, df2, df3, df4, labels, r = prepdf(demgla, dh[i], pathdh, period)
        print('resolution of the dh image: ', r)
        # print('The original dataframe: ', df1)
        # print('The original dataframe with outlier filled: ', df3)
        H,P,A,V = mbmean(df1,df2,df3,df4, dh[i],r)
        S.append(dem_s[:8])
        E.append(dem_e[i][40:48])
        MB.append(H)
        PL.append(P)
        volM.append(V)
        Area.append(A)
        Hpoly, Vpoly = mbpoly(df1,df2,df3,labels,dh[i],pathdh, r)
        MBP.append(Hpoly)
        volP.append(Vpoly)
        # calculate mass balance uncertainty
        L = 500
        SM.append(mb_uncertainty(pathdh + dh[i], pathMask + outMask, H, A, P, V, L))
        SP.append(mb_uncertainty(pathdh + dh[i], pathMask + outMask, Hpoly, A, P, Vpoly, L))
        
    # prepare a datafarme to write results
    df = pd.DataFrame({'Start':S, 'End':E, 'dh_mean_EB':MB, 'dh_polyfit_mean': MBP, 
                       'Area':Area, 'P':PL, 'VolM':volM, 'volP':volP, 'SM':SM, 'SP':SP})
    df['MBmean'] = df['dh_mean_EB']*0.85
    df['MBpoly'] = df['dh_polyfit_mean']*0.85
    yearS = int(dem_s[:4])
    dy = int(yearE) - yearS
    df['MBM'] = df['MBmean']/dy
    df['MBP'] = df['MBpoly']/dy
    df['UM'] = df['SM']/dy
    df['UP'] = df['SP']/dy
    df.to_csv(pathdh + yearE + '_' + dem_s[:4] + '_mb.csv', index=False)
    
    
   