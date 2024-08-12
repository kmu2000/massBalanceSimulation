# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 09:09:16 2023

@author: Kriti Mukherjee
This script can be used to prepare glacier bed using the thickness grid provided 
by Farinotti et al., 2019
SRTM, thickness grid, first surface and glacier polygon are in the same coordinate system
"""

import rasterio as rio
import os
import numpy as np


name = 'RGI15-02999'
path = '../' + name + '/Bed/'
projected = 'EPSG:32645'


surface = '20000211_NASA.tif'
# surface_utm = 'utm_' + surface
thick = 'RGI60-15.02999_thickness.tif'



# glacier polygon and surface for the starting year
glapoly = '/mnt/e/lidar3/SnowModel/' + name + '/extents/extent_rgi_1502999_1969.shp'

surface1 = '19690930_KH4_WestSikkim_coregNASADEM_30m.tif'

# outputs
gla_surface = name + '_' + surface
gla_thick = 'final_' + thick
dataBed = 'Bed_' + name + '_srtm.tif'
dataclass = name + '_class1.tif'
gla_surface1 = name + '_surface1.tif'
mask = name + '_mask.tif'
mask1 = 'final_' + mask



### extract the glacier area use the following lat/lon corners
longitude_upper_left = '88.345'
latitude_upper_left = '27.745'
longitude_lower_right = '88.375'
latitude_lower_right = '27.72' 


# extract glacier area with above extent from the srtm dem
os.system('gdalwarp -overwrite ' + ' -t_srs ' + projected + ' -te ' + longitude_upper_left 
          + ' ' + latitude_lower_right + ' ' + longitude_lower_right + ' ' + latitude_upper_left +  
          ' -te_srs EPSG:4326 -tr 30 30 ' + ' -r average -dstnodata -9999 ' + 
          path + surface + ' ' + path + gla_surface)


# extract glacier area with above extent from the thickness
os.system('gdalwarp -overwrite ' + ' -te ' + longitude_upper_left 
          + ' ' + latitude_lower_right + ' ' + longitude_lower_right + ' ' + latitude_upper_left +  
          ' -te_srs EPSG:4326 -tr 30 30 ' + ' -r average -srcnodata 0 -dstnodata -9999 ' + 
          path + thick + ' ' + path + gla_thick)


# extract glacier area with above extent from the first dem
os.system('gdalwarp -overwrite ' + ' -t_srs ' + projected + ' -te ' + longitude_upper_left 
          + ' ' + latitude_lower_right + ' ' + longitude_lower_right + ' ' + latitude_upper_left +  
          ' -te_srs EPSG:4326 -tr 30 30 ' + ' -r average -dstnodata -9999 ' + 
          path + surface1 + ' ' + path + gla_surface1)


# rasterize glacier polygon to create a class image for the first year
os.system('gdal_rasterize -burn 1 -tr 30 30 ' + glapoly + ' ' + path + mask)


# extract glacier area with above extent from the class mask
os.system('gdalwarp -overwrite ' + ' -t_srs ' + projected + ' -te ' + longitude_upper_left 
          + ' ' + latitude_lower_right + ' ' + longitude_lower_right + ' ' + latitude_upper_left +  
          ' -te_srs EPSG:4326 -tr 30 30 ' + ' -r average -dstnodata -9999 ' + 
          path + mask + ' ' + path + mask1)

src = rio.open(path+gla_surface)

# read the metadata for the output geotifs
profile = src.profile
profile.update(dtype=rio.float32,count=1)

# create the bed by subtracting thickness pixels from the DEM and setting 0 values to elevation
dataEle = rio.open(path + gla_surface).read(1)
dataThick =  rio.open(path+gla_thick).read(1)
datamask1 = rio.open(path + mask1).read(1)

print('shape of thickness data: ', dataThick.shape)
print('shape of elevation data: ', dataEle.shape)
print('shape of mask data: ', datamask1.shape)

# create the Bed and glacier class data
Bed = np.where(dataThick>0,dataEle-dataThick,dataEle)


# create the first surface
sf1 = rio.open(path + gla_surface1, 'r').read(1)
data_sf1 = np.where(datamask1==1, sf1, dataEle)
data_th1 = data_sf1-Bed
data_sf1 = np.where(data_th1<0, dataEle, data_sf1)
classGla = np.where(data_th1>0,20,3)
print('shape of first surface data: ', data_sf1.shape)


# write the bed, class and surface output
with rio.open(path + dataBed, 'w', **profile) as dst:
    dst.write(Bed.astype(rio.float32),1) 

profile.update(dtype=rio.int32,count=1,nodata=-9999)
with rio.open(path + dataclass, 'w', **profile) as dst:
    dst.write(classGla.astype(rio.int32),1) 

profile.update(dtype=rio.float32,count=1,nodata=-9999)    
with rio.open(path + gla_surface1, 'w', **profile) as dst:
    dst.write(data_sf1.astype(rio.float32),1) 

