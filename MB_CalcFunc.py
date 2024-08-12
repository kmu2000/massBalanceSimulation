# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 08:11:12 2023

@author: Kriti.Mukherjee
"""

import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os


def elebinprep(elemin, elemax):
    nmin = elemin//50
    nmax = elemax//50
    
    minEle = nmin*50
    maxEle = 50 + nmax*50
    bins = range(int(minEle), int(maxEle), 50)
    labels = range(int(minEle + 25), int(maxEle-25), 50)
    return bins, labels 


def prepdf(dem, dh, path, period):
    outplot = path + 'plots/'
    if not os.path.exists(outplot):
        os.makedirs(outplot)
    # read the dem and dh files
    srcdem = rasterio.open(path + dem)
    srcdh = rasterio.open(path + dh)
    
    gt = srcdh.res
    # print(dir(src))
    res = gt[1]

    # read the metadata to write output file
    profile = srcdem.profile
    profile.update(dtype=rasterio.float32, count=1)

    # read the dh and dem as array
    demdata = srcdem.read(1)
    dhdata = srcdh.read(1)
    # get the dimension of the arrays
    size = demdata.shape

    # array to vector
    DEM_V = np.ravel(demdata)
    DEM_V = np.where(DEM_V <0, np.nan, DEM_V)
    DH_V = np.ravel(dhdata)
    DH_V[abs(DH_V)>100] = np.nan

    # find the minimum and maximum elevation
    Emin = np.nanmin(DEM_V)
    Emax = np.nanmax(DEM_V)

    bins, labels = elebinprep(Emin, Emax)

    # create pandas dataframe for elevation, and dh
    df1 = pd.DataFrame({"Z":DEM_V, 'DH':DH_V})
    # find the median elevation
    ME = df1['Z'].median()
    
    # classify the elevation data into 50 metre bins and add to the data frame
    df1['Zbins'] = pd.cut(df1['Z'], bins, labels=labels) # labels are the mid points of the elevation bins
    #print(df1['Zbins'].unique())
    dem_class = df1['Zbins'].to_numpy()
    class_array = np.reshape(dem_class,(size[0],size[1]))
    
    
    # plot the classified elevation bins data
    fig, ax = plt.subplots(figsize=(7,9))
    im = ax.imshow(class_array, cmap='terrain')
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    fig.colorbar(im, cax=cax)
    ax.set_title("DEM classified by 50 m elevation band")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(outplot + 'Elebands' + dh[:-4] + '.png', dpi=600)
        
    
    # calculate mean and standard deviation of each elevation band 
    df2 = df1.groupby(['Zbins']).mean()
    df3 = df1.groupby(['Zbins']).std()
    dfmin = df1.groupby(['Zbins']).min()
    dfmax = df1.groupby(['Zbins']).max()
    MeanDH = df2.DH.values
    StdDH = df3.DH.values
    UDH1 = MeanDH + 3*StdDH
    LDH1 = MeanDH - 3*StdDH
    UDH2 = MeanDH + StdDH
    LDH2 = MeanDH - StdDH
    
    # dataframe showing mean and standard deviation of each elvation band 
    dfStat = pd.DataFrame({'Zbins':df2.reset_index().Zbins.values,'MeanDH':MeanDH,
                           'StdDH':StdDH, 'UDH1':UDH1,'LDH1':LDH1,'UDH2':UDH2,'LDH2':LDH2})

    dfStatI = dfStat.set_index('Zbins')
    # print(dfStatI)
    
    for i in labels:
        if i<ME:
            UL = dfStat.loc[dfStat['Zbins']==i,'UDH1'].values[0]
            LL = dfStat.loc[dfStat['Zbins']==i,'LDH1'].values[0]
            # print(i,UL,LL)
            df1['DHF'+str(i)]=np.where(((df1['Zbins']==i) & ((df1['DH']>LL) & (df1['DH']<UL))),df1['DH'],np.nan)
        else:
            UL = dfStat.loc[dfStat['Zbins']==i,'UDH2'].values[0]
            LL = dfStat.loc[dfStat['Zbins']==i,'LDH2'].values[0]
            # print(i,UL,LL)
            df1['DHF'+str(i)]=np.where(((df1['Zbins']==i) & ((df1['DH']>LL) & (df1['DH']<UL))),df1['DH'],np.nan)

    dfData = df1[['DHF'+str(i) for i in labels]]
    dfData['DHF']= dfData.sum(axis=1, min_count=1)
    # print(dfData.head())
    
    # merge the final outlier removed column with the existing df
    dfMerge = pd.concat((df1[['Zbins','Z','DH']],dfData['DHF']), axis=1)
    

    # count the number of pixels in the DEM, in dh before and after outlier removal
    PixelCount = dfMerge.groupby('Zbins').count()
    
    # find the mean dh before (dfMean['DH']) and after (dfMean['DHF']) outlier removal
    Mean = dfMerge.groupby('Zbins').mean() 
    dfMB = pd.concat((PixelCount[['Z']],Mean[['DH','DHF']]), axis=1)
    print('shape of merged data after removing outlier: ', dfMerge.shape)
    #percent glacier area = number of pixel in ele band/ total number
    dfMB['PA_EB']=(dfMB['Z']/dfMB['Z'].sum())
    # volume change per ele band = number of pixel*mean elevation change 
    dfMB['VDHF'] = dfMB['Z']*dfMB['DHF']
    
    
    # outlier removed dh
    dh_outrem = dfData['DHF'].to_numpy()
    dh_arrayF = np.reshape(dh_outrem,(size[0],size[1]))
    
    # outlier removed annual dh
    dh_outrem = dfData['DHF'].to_numpy()/period
    dh_outremF = np.reshape(dh_outrem,(size[0],size[1]))
    
    # void filled dh
    dh_filled = dfMerge.merge(dfMB, on='Zbins', how='outer')
    print('dh data after merging : ', dh_filled)
    # # Replace 'nodata' with values from replacement_column where available
    # dh_filled['Filled'] = dh_filled.apply(
    # lambda row: row['DHF_y'] if row['DHF_x'] == 'nodata' else row['DHF_x'],
    # axis=1)

    # writing the outlier removed dh image 
    with rasterio.open(outplot + 'outrem_' + dh, 'w', **profile) as dst:
        dst.write(dh_arrayF.astype(rasterio.float32),1)
        
    # writing the outlier removed annual dh image 
    with rasterio.open(outplot + 'outremA_' + dh, 'w', **profile) as dst:
        dst.write(dh_outremF.astype(rasterio.float32),1)
        
    # void filled dh write
    dh_voidFill = dh_filled['DHF_y'].to_numpy()
    dh_arrayFill = np.reshape(dh_voidFill,(size[0],size[1]))
    
    # writing the void filled dh image 
    with rasterio.open(outplot + 'voidfillMean_' + dh, 'w', **profile) as dst:
        dst.write(dh_arrayFill.astype(rasterio.float32),1)
    
    return df1, dfMB, dh_filled, dfData, labels, res

def mbmean(df1, dfMB, dh_filled, dfData, dh, res):
    
    DhGlaB =  (dfMB['Z']*dfMB['DH']).sum()/dfMB['Z'].sum()
    # thickness change is the total volume change for all pixels divided by total number of pixels
    delZ = dfMB['VDHF'].sum()/dfMB['Z'].sum()
    delV = (dfMB['VDHF']).sum()*res*res 
    # print(dfMB)
    # print('dh of glacier before outlier removal = ', DhGlaB)
    # print('dh of glacier after outlier removal = ', delZ)
    # print('Volume change of glacier after outlier removal = ', delV)
    # print('Mass change of glacier after outlier removal =', delV*850)
    
    # Plot the dh vs elevation bin before and after outlier removal
    # fig3,ax = plt.subplots()
    # dfMB.reset_index().plot(x='Zbins',y='DH', ax=ax, color='r')
    # dfMB.reset_index().plot(x='Zbins',y='DHF', ax=ax, color='g')
    # plt.savefig(path + dh[:-4] + '.png')
    # count the number of pixels
    total = df1['Z'].count()
    data = df1['DH'].count()
    dataF = dfData['DHF'].count()
    Area = total*res*res # where 3(m) is the patial resolution of the DEM
    p = dataF/total
    # print("Total data points = ", total)
    # print("Glacier Area = ", Area)
    # print("percent before outlier removal = ", data/total)
    # print("percent after outlier removal = ", p)

    # ◙print('all done')
    
    return delZ, p, Area, delV
    

def mbpoly(df1, dfMB, dfData, labels, dh, path, r):
    outplot = path + 'plots/'
    if not os.path.exists(outplot):
        os.makedirs(outplot)
    x = [i for i in labels]
    y = dfMB.DHF.fillna(0).values
    
    import numpy.polynomial.polynomial as poly
    coefs = poly.polyfit(x, y, 3)    
    ffit = poly.polyval(x, coefs)
    

    dfMB['DHF1'] = ffit
    # print(dfMB)

    # Plot the dh vs elevation bin before and after outlier removal
    fig,ax = plt.subplots()
    ax.scatter(dfMB.reset_index().Zbins,dfMB.DH, color='r', label='before outlier remove')
    ax.scatter(dfMB.reset_index().Zbins,dfMB.DHF, color='g', label='after outlier remove')
    ax.set_xlabel("Elevation (m)")
    ax.set_ylabel("mean thickness change (m)")
    ax.plot(x, ffit, color='k', label='polynomial fit')
    ax.legend()
    plt.savefig(outplot + 'polyfit_' + dh[:-4] + '.png', dpi=600)

    # mass balance and volume calculation
    DhGlaB =  (dfMB['Z']*dfMB['DH']).sum()/dfMB['Z'].sum()
    delZ = (dfMB['Z']*dfMB['DHF1']).sum()/dfMB['Z'].sum()
    delV = (dfMB['Z']*dfMB['DHF1']*r*r).sum() 
    #print(dfMB)
    # print('dh of glacier before outlier removal = ', DhGlaB)
    # print('dh of glacier after outlier removal = ', delZ)
    # print('dh of glacier after polyfit = ', delZ1)

    # print('Volume change of glacier after outlier removal = ', delV)
    # print('Mass change of glacier after outlier removal =', delV*850)
    # print('Mass change in Gt of glacier after outlier removal =', delV*850/10**(12))
    # print('Mass change in Kt of glacier after outlier removal =', delV*850/10**(6))
    return delZ, delV


def mb_uncertainty(dh, Mask, delZ, Area, p, delV, L=500):
    dh_A = rasterio.open(dh).read(1)
    Mask_A = rasterio.open(Mask).read(1)
    #dem_A = gdal.Open(dem).ReadAsArray().astype(np.int)
    Mask_A = np.where(Mask_A <0, np.nan, Mask_A)
    dh_A[abs(dh_A)>30]=np.nan

    # get the rows and columns
    rows = dh_A.shape[0]
    cols = dh_A.shape[1]

    # create vectors
    dh_data = np.ravel(dh_A)
    Mask_data = np.ravel(Mask_A)


    # create dataframe with dh and mask data
    df = pd.DataFrame({'dh': dh_data, 'Mask': Mask_data})
    #print(df.Mask.min(),df.Mask.max())


    df['stable_dh'] = np.where((df['Mask']==0),df['dh'],np.nan)

    dh_st = np.reshape(df.stable_dh.values,(rows,cols))

    plt.imshow(dh_st, vmin=-10, vmax=10, cmap='RdBu')

    st_dh = df['stable_dh'].std()
    mean_dh =  df['stable_dh'].mean()
    min_dh =  df['stable_dh'].min()
    max_dh =  df['stable_dh'].max()

    print(mean_dh, st_dh, min_dh,max_dh)

    Ac = math.pi*L*L
    # Area = 12531465
    sigmadelZ = st_dh*np.sqrt(Ac/(5*Area))
    sigmaA = 0.1*Area

    print('sigmadelZ: ', sigmadelZ)
    print('sigmaA: ', sigmaA)

    sigmadelV = np.sqrt((sigmadelZ*(p+5*(1-p))*Area)**2+(sigmaA*delZ)**2)
    print("sigmadelV : ", sigmadelV)
    fdelV = 850
    sigmafdelV = 60


    sigmaM = np.sqrt((sigmadelV*fdelV)**2+(sigmafdelV*delV)**2)    
    return sigmaM/(Area*1000) # convertion of kg to m w.e.

path = '/mnt/e/Glaciology/MassBalance/DH/'
dh = 'Chhota_dh_2020_2009.tif'
dem = 'Chhota_n32_e077_3arc_v2_32643.tif'

if __name__=='__main__':
    df1, df2, df3 = prepdf(dem, dh, path)
    M, P = mbmean(df1,df2,df3)














