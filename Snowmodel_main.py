# This script can be used to run snowmodel coupled with ice dynamics model
# We need to prepare the folders for each year run and the ./snowmodel file before running it 
from __future__ import division
import glob
import os
import time as time
import numpy as np
# from netCDF4 import Dataset as dt
from Snowmodel_functions import build_index_arrays, step, SM_out, grads_nc, nc_MB, set_ele, prepare_parfile
from tiftodat import tif_dat
import constants
import rasterio as rasterio
from datetime import date
import pandas as pd
import shutil

def convert(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))

start = time.time()
name = '15_02999' # name of the glacier
StnEle = 5210

path = '/mnt/e/lidar3/SnowModel/RGI15-02999/Farinotti2019/SMID/'
SM_out_path = path + 'outputs/'
if not os.path.exists(SM_out_path):
    os.makedirs(SM_out_path)


nx = '98'
ny = '93'
res = '30.0'
xmn = '632591.001882813987'
ymn = '3066922.340428806841'

year_start = 1969
year_end = 2022

lat = 27.7

runs = []
years = np.arange(year_start,year_end+1)
me = 9
de = 30

days = []
hours = []
for i in range(len(years)-1):
#     print(years[i])
    dt1 = date(years[i],me,de)
    dt2 = date(years[i+1],me,de)
    ds = (dt2-dt1).days
    days.append(ds)
    hours.append(ds*24)

print(hours)
print(years)

daystart = 0
daywinter = 241  # considering winter from October to May next year
for i in range(year_start,year_end+1):
    runs.append(str(i)[-2:] + '-' + str(i+1)[-2:])
    
print(runs)

for i in range(len(runs)-1):
    # change met file elevation
    set_ele(runs[i], StnEle, path)
    
    #prepare snowmodel.par file
    path_run = path + runs[i] + '/'
    print(path_run)
    par_in = 'snowmodel.par'
    ctl_in = 'sp_1986_1987.ctl'
    ctl_in2 = 'sp_'+str(years[i])+'_'+str(years[i+1])+'.ctl'

    f1 = open(path + par_in, 'r')
    f2 = open(path_run + par_in, 'w')
    
    prepare_parfile(nx,ny,res, xmn, ymn, lat, hours[i], par_in, ctl_in, years[i], runs[i], path, path_run)
        
    os.chdir(path + runs[i] + '/code')
    os.system('./compile_snowmodel.script')
            
    # create the ctl files
    f3 = open(path + ctl_in, 'r')
    f4 = open(SM_out_path + ctl_in2, 'w')
    for line in f3:
        if line.startswith('DSET'):
            line = 'DSET  ^sp' + runs[i] + '.gdat'
            f4.write(line)
            f4.write('\n')
        elif line.startswith('XDEF'):
            line = 'XDEF   ' + nx + ' LINEAR 0 0.05'
            f4.write(line)
            f4.write('\n')
        elif line.startswith('YDEF'):
            line = 'YDEF   ' + ny + ' LINEAR 0 0.05'
            f4.write(line)
            f4.write('\n')
        elif line.startswith('TDEF'):
            line = 'TDEF  ' + str(days[i]) + ' LINEAR 10Z1oct' + str(years[i]) + ' 1dy'
            f4.write(line)
            f4.write('\n')
        else:
            f4.write(line)
    f3.close()
    f4.close()

ctlName = []
nc_files = []
f_in_glacier = []
f_in_surface = []

for ctlfile in glob.glob(os.path.join(SM_out_path, '*.ctl')):
    ctlName.append(ctlfile[-13:-4])
    f_in_glacier.append(name + '_' + ctlfile[-8:-4] + '.tif')
    f_in_surface.append('SA_' + ctlfile[-8:-4] + '.tif')

ctlName.sort()

for i in range(len(ctlName)):
    nc_files.append(name + '_' +  ctlName[i] + '.nc')
nc_files.sort()
# print(nc_files)
f_in_glacier.sort()

f_in_surface.sort()
# print(f_in_glacier)
# print(f_in_surface)

grads_file = glob.glob(os.path.join(SM_out_path, '*.ctl'))
grads_file.sort()
# print(grads_file)

MByear = []
MBimage = []
MBwinter = []
MBsummer = []
meanMB = []
medianMB = []
MByear_gla = []


for i in nc_files:
    MByear.append(i[-13:-3])
    MBimage.append('MB'+i[-13:-3]+'.tif')
    MBwinter.append('MB-winter'+i[-13:-3]+'.tif')
    MBsummer.append('MB-summer'+i[-13:-3]+'.tif')
    MByear_gla.append('MB_glacier'+i[-13:-3]+'.tif')


args = ('./snowmodel')
for i in range(len(runs)-1):
    topo_veg_inpath = path + runs[i] + '/' + 'topo_veg/'
    topo_veg_outpath = path + runs[i+1] + '/' + 'topo_veg/'
    class1 = rasterio.open(path + runs[i]+ '/topo_veg/class_' + str(years[i]) + '.tif')
    glaclass = class1.read(1)
    
    if not os.path.exists(topo_veg_outpath):
        os.makedirs(topo_veg_outpath)
    print('Starting snowmodel simulation for hydrologic year', runs[i], 'now...')
    SM_out(args,path + runs[i] +'/')
    grads_nc(grads_file[i],SM_out_path, nc_files[i])
    meanA, medianA = nc_MB(SM_out_path,nc_files[i], days[i], MBimage[i], glaclass)
    meanW, medianW = nc_MB(SM_out_path,nc_files[i],daywinter+1,MBwinter[i], glaclass)
    meanMB.append(meanA)
    medianMB.append(medianA)
    os.system('gdal_calc.py -A ' + SM_out_path + MBimage[i] + ' -B ' + SM_out_path + MBwinter[i] +
              ' --calc="(A-B)" ' + '--outfile=' + SM_out_path + MBsummer[i])  
    
    
    f_in_bdot = SM_out_path + MBimage[i]
    in_surface = topo_veg_inpath + 'surface_' + str(years[i]) + '.tif'
    data = rasterio.open(in_surface)
    surface = data.read(1)
    data = rasterio.open(f_in_bdot)
    b_dot = data.read(1)
    out_surface = 'surface_' + str(years[i+1]) + '.tif'

    out_glacier = 'class_' + str(years[i+1]) + '.tif'

    S = surface.ravel()
    B = constants.bed.ravel()   
    
    # thickness
    H = S - B
    H_old = S - B

    ice_mask = S - B
    
    ice_mask[ice_mask <= constants.ice_h_min] = 3.0
    ice_mask[ice_mask > constants.ice_h_min] = 20.0
    
    # print('ice class min and max: ', np.min(ice_mask), np.max(ice_mask))
    ## flag for conjugate gradient solver
    flag = 1

    ## create empty lists to write to
    Times = []
    G_wastage = []
    G_melt = []
    G_area = []

    ## define start end times and dt
    dt = 1 / 12.  ##monthly time step
    t_STOP = 1.0
    t_START = 0.0
    t = t_START

    ## reshape b_dot to conform with other 1D vectors for linear algebra work
    b_dot = np.reshape(b_dot, (len(b_dot.ravel()), 1))

    ic_jc, ip_jc, im_jc, ic_jp, ic_jm, ip_jm, im_jm, im_jp = build_index_arrays(constants.ny, constants.nx)

    for yr in np.arange(t_STOP / dt):
        S, t, H_old, Q_wast, IA, flag = step(B, S, b_dot, dt, t, flag, H_old, ic_jc, ip_jc, im_jc, ic_jp, ic_jm, ip_jm,
                                             im_jm, im_jp)
        bs = b_dot[:, 0]
        ind = np.nonzero(bs[H_old > 0])  # get indices for bdot where there is ice
        g_melt = (np.minimum((bs[ind] + H_old[ind]), 0))  # minimum of (bdot + ice thickness), zero
        g_melt = - g_melt.sum() * (1 / constants.sec_yr) * dt * constants.dx * constants.dy * (constants.RHO / 1000.0)
        # g_melt                                  = -g_melt.sum()*dt*dx*dx*dy*(RHO/1000.0)
        Times.append(t)
        G_wastage.append(Q_wast)
        G_melt.append(g_melt)
        G_area.append(IA)
        
        
    new_surface = np.reshape(S, (constants.ny, constants.nx))
    new_glacier = np.reshape(ice_mask, (constants.ny, constants.nx))
    
    # writing the filled dem
    profile = constants.profile
    profile.update(dtype=rasterio.float32,count=1,nodata=-9999)
    with rasterio.open(topo_veg_outpath + out_surface, 'w', **profile) as dst:
        dst.write(new_surface.astype(rasterio.float32),1)
    with rasterio.open(topo_veg_outpath + out_glacier, 'w', **profile) as dst:
        dst.write(new_glacier.astype(rasterio.float32),1)
    
           
    os.chdir(topo_veg_outpath)
    tif_dat(out_glacier, topo_veg_outpath)
    tif_dat(out_surface, topo_veg_outpath)
           


print(meanMB, medianMB)

MBT_mean = sum(meanMB)
MBT_median = sum(medianMB)

print('MBT_mean: ', MBT_mean)
print('MBT_median: ', MBT_median)

MByear = [i+1 for i in years[:-1]]  
df = pd.DataFrame({'Year':MByear, 'MBmean': meanMB, 'MBmedian':medianMB})
df.to_csv(SM_out_path + 'MBseries.csv', index=False) 

# calibration MB inputs
start_cal = 1984
end_cal = 2022
MB_cal = -28.3
MB_U = 0.5
df['MBC'] = df['MBmean'].cumsum()
MB_M = df[df['Year']==end_cal]['MBC'].values[0] - df[df['Year']==start_cal]['MBC'].values[0]
diff = MB_cal - MB_M
print('cumulative modelled mass balance for comparison: ', MB_M)
print('difference of modelled and geodetic mass balance: ', diff)

# decide whether stn elevation should be increased or decreased for calibration
if diff>MB_U:
    print('modelled mass balance is more negative, please decrease station elevation than ', StnEle)
elif diff<-MB_U:
    print('modelled mass balance is more positive, please increase station elevation than ', StnEle)
else:
    print('modelled elevation: ', StnEle)

# remove all gdat and nc files
filenc = glob.glob(SM_out_path + '*.nc')
filegdat = glob.glob(SM_out_path + '*.gdat')

for nc,gdat in list(zip(filenc,filegdat)):
    os.remove(nc)
    os.remove(gdat)


end = time.time()
timeTaken = (end-start)
print("time taken: ", convert(timeTaken))
