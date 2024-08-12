from __future__ import division
import glob
import os
import time as time
import numpy as np
from Snowmodel_functions import SM_out, grads_nc, nc_MB, set_ele
import rasterio as rasterio
import multiprocessing
from datetime import date
import pandas as pd


def convert(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))

start = time.time()
name = '15_02999' # name for the nc file
StnEle = 5120

path = '/mnt/e/lidar3/SnowModel/RGI15-02999/Farinotti2019/SMWO/'

SM_out_path = path + 'outputs/'
if not os.path.exists(SM_out_path):
    os.makedirs(SM_out_path)

nx = '98'
ny = '93'
res = '30.0'
xmn = '632591.001882813987'
ymn = '3066922.340428806841'

surface_start = 'surface_1969.dat'
class_start = 'class_1969.dat' 

year_start = 1969
year_end = 1983


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
daystart = 0
daywinter = 241
for i in range(year_start,year_end):
    runs.append(str(i)[-2:] + '-' + str(i+1)[-2:])


print('runs: ', runs)    
for i in range(len(runs)):
    # change met file elevation
    set_ele(runs[i], StnEle, path)
    
    #prepare snowmodel.par file and ctl file
    path_run = path + runs[i] + '/'
    print(path_run)
    par_in = 'snowmodel.par'
    ctl_in = 'sp_1986_1987.ctl'
    ctl_in2 = 'sp_'+str(years[i])+'_'+str(years[i+1])+'.ctl'

    f1 = open(path + par_in, 'r')
    f2 = open(path_run + par_in, 'w')
    for line in f1:
        if line[6:14].startswith('nx'):
            line = 'nx = ' + nx
            f2.write(line)
            f2.write('\n')
        elif line[6:14].startswith('ny'):
            line = 'ny = ' + ny
            f2.write(line)
            f2.write('\n')
        elif line[6:14].startswith('deltax'):
            line = 'deltax = ' + res
            f2.write(line)
            f2.write('\n')
        elif line[6:14].startswith('deltay'):
            line = 'deltay = ' + res
            f2.write(line)
            f2.write('\n')
        elif line[6:14].startswith('xmn'):
            line = 'xmn = ' + xmn
            f2.write(line)
            f2.write('\n')
        elif line[6:14].startswith('ymn'):
            line = 'ymn = ' + ymn
            f2.write(line)
            f2.write('\n')
        elif line[6:23].startswith('iyear'):
            line = 'iyear_init = ' + str(years[i])
            f2.write(line)
            f2.write('\n')
        elif line[6:21] == 'max_iter = 8760':
            line = 'max_iter = ' + str(hours[i])
            f2.write(line)
            f2.write('\n')
        elif line[6:53].startswith('met'):
            line = 'met_input_fname = met/met_data_' + runs[i] + '.dat'
            f2.write(line)
            f2.write('\n')
        
        elif line[6:53].startswith('snowpack'):
            #print(line)
            line = 'snowpack_output_fname = ../outputs/sp' + runs[i] +'.gdat'
            f2.write(line)
            f2.write('\n')
        
        elif line[0:27] == 'topo_ascii_fname = topo_veg':
            #print(line)
            line = 'topo_ascii_fname = topo_veg/' + surface_start
            f2.write(line)
            f2.write('\n')
        elif line[0:26] == 'veg_ascii_fname = topo_veg':
            line = 'veg_ascii_fname = topo_veg/' + class_start
            f2.write(line)
            f2.write('\n')
        else:
            f2.write(line)
    f1.close()
    f2.close()
    
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

for ctlfile in glob.glob(os.path.join(SM_out_path, '*.ctl')):
    ctlName.append(ctlfile[-13:-4])

ctlName.sort()
print(ctlName)


for i in range(len(ctlName)):
    nc_files.append(name + '_' +  ctlName[i] + '.nc')
nc_files.sort()
print(nc_files)


grads_file = glob.glob(os.path.join(SM_out_path, '*.ctl'))
grads_file.sort()
print(grads_file)


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

execFile = ('./snowmodel')

# Run snowmodels of different years in multiple cores simultaneously
if __name__ == '__main__':
    #startime = time.time()
    processes = []
    for i in range(len(runs)):
        p = multiprocessing.Process(target=SM_out, args=(execFile,path + runs[i] +'/'))
        processes.append(p)
        p.start()

    for process in processes:
        process.join()

for i in range(len(runs)):
    class1 = rasterio.open(path + runs[i]+ '/topo_veg/' + class_start) # change the name of the class data
    glaclass = class1.read(1)
    grads_nc(grads_file[i],SM_out_path, nc_files[i])
    MB_mean, MB_median = nc_MB(SM_out_path,nc_files[i], days[i], MBimage[i], glaclass)
    meanMB.append(MB_mean)
    medianMB.append(MB_median)
    meanW, medianW = nc_MB(SM_out_path,nc_files[i],daywinter+1,MBwinter[i], glaclass)
    os.system('gdal_calc.py -A ' + SM_out_path + MBimage[i] + ' -B ' + SM_out_path + MBwinter[i] +
              ' --calc="(A-B)" ' + '--outfile=' + SM_out_path + MBsummer[i])  

MBT_mean = sum(meanMB)
MBT_median = sum(medianMB)

print('MB_mean: ', meanMB)
print('MB_median: ', medianMB)
print('MBT_mean: ', MBT_mean)
print('MBT_median: ', MBT_median)

MByear = [i+1 for i in years[:-1]]    
df = pd.DataFrame({'Year':MByear, 'MBmean': meanMB, 'MBmedian':medianMB})
df.to_csv(SM_out_path + 'MBseries.csv', index=False) 


# df = pd.read_csv(SM_out_path + 'MBseries.csv')
# calibration MB inputs
start_cal = 1970
end_cal = 1983
MB_cal = 9.8
MB_U = 0.5
df['MBC'] = df['MBmean'].cumsum()
MB_M = df[df['Year']==end_cal]['MBC'].values[0]
diff = MB_cal - MB_M
print('cumulative modelled mass balance for comparison: ', MB_M)
print('difference of modelled and geodetic mass balance: ', diff)

if diff>0.1:
    print('modelled mass balance is more negative, please decrease station elevation than ', StnEle)
elif diff<-0.1:
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
print('All done!')

