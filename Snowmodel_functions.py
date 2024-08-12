import subprocess
import glob
import os
from osgeo import gdal
import numpy as np
from netCDF4 import Dataset as dt
from numba import jit
import constants
import numexpr as ne
from scipy import sparse, linalg
from scipy.sparse.linalg import cg
from pathlib import Path
import pandas as pd


def prepare_parfile(nx, ny, res, xmn, ymn, lat, hour, inpar, inctl, year, run, albedo, path, outpath):
    f1 = open(path + inpar, 'r')
    f2 = open(outpath + inpar, 'w')    
    
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
        elif line[6:10].startswith('xlat'):
            line = 'xlat = ' + str(lat)
            f2.write(line)
            f2.write('\n')
        elif line[6:23].startswith('iyear'):
            line = 'iyear_init = ' + str(year)
            f2.write(line)
            f2.write('\n')
        elif line[6:20].startswith('albedo_glacier'):
            line = 'albedo_glacier = ' + str(albedo)
            f2.write(line)
            f2.write('\n')
        elif line[6:21] == 'max_iter = 8760':
            line = 'max_iter = ' + str(hour)
            f2.write(line)
            f2.write('\n')
        elif line[6:53].startswith('met'):
            line = 'met_input_fname = met/met_data_' + run + '.dat'
            f2.write(line)
            f2.write('\n')
        
        elif line[6:53].startswith('snowpack'):
            #print(line)
            line = 'snowpack_output_fname = ../outputs/sp' + run +'.gdat'
            f2.write(line)
            f2.write('\n')
        
        elif line[0:27] == 'topo_ascii_fname = topo_veg':
            #print(line)
            line = 'topo_ascii_fname = topo_veg/' + 'surface_' + str(year) + '.dat' 
            f2.write(line)
            f2.write('\n')
        elif line[0:26] == 'veg_ascii_fname = topo_veg':
            line = 'veg_ascii_fname = topo_veg/' + 'class_' + str(year) + '.dat'
            f2.write(line)
            f2.write('\n')
        else:
            f2.write(line)
    f1.close()
    f2.close()
    


def set_ele(run, E, path):
    fields = ['year','month','day','hour','stnID','E','N','Ele','T','RH','WS','WD','PPT'] 
    df = pd.read_csv(path + run + '/met/' + 'met_data_' + run + '.dat', names=fields,sep='\t')
    df.Ele = E
    df.to_csv(path + run + '/met/' + 'met_data_' + run + '.dat', header=False, sep='\t', index=False)
 

def SM_out(args,path):
    p = subprocess.Popen(args, stdout=subprocess.PIPE, bufsize=1, cwd=path)
    for line in iter(p.stdout.readline, b''):
        print(line)
    p.stdout.close()
    p.wait()


def grads_nc(gdat_file,ctl_path,netcdf_file):
    args = ('/usr/bin/cdo', '-f', 'nc', 'import_binary', gdat_file, ctl_path+netcdf_file)
    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    popen.wait()
    output = popen.stdout.read()

def nc_MB(nc_path, nc_file, days, MB, classdata):
    ncfile = dt(nc_path + nc_file,'r')

    R       = ncfile['sumroff'][days-1,0,:,:]
    P       = ncfile['sumsprec'][days-1,0,:,:]
    S       = ncfile['sumsublim'][days-1,0,:,:]

    B      =  (P - S - R)

    Bn = np.empty((0,constants.imagesize[1]))
    for k in range(constants.imagesize[0]):
        Bn = np.append(Bn,[B[constants.imagesize[0]-1-k,:]], axis=0)
    Bf = np.where(classdata==20, Bn, 0)
    Bc = np.where(classdata==20, Bn, np.nan)
    output = gdal.GetDriverByName('GTiff').Create(nc_path + MB, constants.imagesize[1], constants.imagesize[0],1,gdal.GDT_Float32)
    output.SetGeoTransform(constants.geo)
    output.SetProjection(constants.proj)
    output.GetRasterBand(1).WriteArray(Bf)
    return np.nanmean(Bc),np.nanmedian(Bc)


def MB_glacier_SM(path, shp, MB, MB_gla):
    
    os.system('gdalwarp -overwrite -cutline' + ' ' + shp + ' ' + path + MB + ' ' + path + MB_gla
          + ' ' + '-dstnodata' + ' ' + '-9999 > /dev/null')
    
    SWE_model = gdal.Open(path + MB_gla).ReadAsArray()

    # convert null values (-9999) to NaN 
    SWE_model[SWE_model == -9999] = np.nan
    # convert 2-d elevation and SWE arrays to 1-D vectors using ravel function
    SWE_modelvec = np.ravel(SWE_model)
    # create pandas dataframe for elevation and SWE
    df = pd.DataFrame({"SWE-model":SWE_modelvec})
    median = df["SWE-model"].median()
    mean = df['SWE-model'].mean()
    
    return mean, median
    

@jit
def build_index_arrays(ny, nx):
    ####################################################
    # function build index arrays

    # create an array of indices for dem
    # incrementing column wise
    # i.e. col 1 = 1,2,3,4, ..., n

    N = ny * nx

    k_ = np.transpose(np.reshape(np.arange(N), (ny, nx)))

    k_ = np.array(k_, dtype=int)

    # Set up generic index arrays (but first set them to empty)

    ic_jc = np.zeros([N, 1], dtype=int)
    ip_jc = np.zeros([N, 1], dtype=int)
    im_jc = np.zeros([N, 1], dtype=int)
    ic_jp = np.zeros([N, 1], dtype=int)
    ic_jm = np.zeros([N, 1], dtype=int)
    ip_jm = np.zeros([N, 1], dtype=int)
    im_jm = np.zeros([N, 1], dtype=int)
    im_jp = np.zeros([N, 1], dtype=int)

    cnt = 0

    for c in range(0, ny):
        for r in range(0, nx):
            ic = c
            jc = r
            ip = min(ic + 1, ny - 1)
            im = max(ic - 1, 0)
            jp = max(r - 1, 0)
            jm = min(r + 1, nx - 1)

            ic_jc[cnt] = k_[jc, ic]
            ip_jc[cnt] = k_[jc, ip]
            im_jc[cnt] = k_[jc, im]
            ic_jp[cnt] = k_[jp, ic]
            ic_jm[cnt] = k_[jm, ic]
            ip_jm[cnt] = k_[jm, ip]
            im_jp[cnt] = k_[jp, im]
            im_jm[cnt] = k_[jm, im]

            cnt += 1

    return (ic_jc, ip_jc, im_jc, ic_jp, ic_jm, ip_jm, im_jm, im_jp)


#ic_jc, ip_jc, im_jc, ic_jp, ic_jm, ip_jm, im_jm, im_jp = build_index_arrays(ny, nx)


## function step
def step(B, S, b_dot, dt, t, flag, H_old,ic_jc, ip_jc, im_jc, ic_jp, ic_jm, ip_jm, im_jm, im_jp):
    A_tilde = constants.A_tilde
    C_tilde = constants.C_tilde
    nm_half = constants.nm_half
    mm_half = constants.mm_half
    np1 = constants.np1
    OMEGA = max(constants.n_GLEN, constants.m_SLIDE) / 2.0
    N = len(S)

    ## impose zero thickness boundary conditions at edge of map domain
    ## keep track of removed ice to add to ice volume
    S = np.reshape(S, (constants.ny, constants.nx))
    B = np.reshape(B, (constants.ny, constants.nx))

    V_t2_edge = (np.ravel(S[0:2, :] - B[0:2, :])).sum() + (np.ravel(S[-2:, :] - B[-2:, :])).sum() + (
    np.ravel(S[:, 0:2] - B[:, 0:2])).sum() + (np.ravel(S[:, -2:] - B[:, -2:])).sum()

    S[0:2, :], S[-2:, :], S[:, 0:2], S[:, -2:] = B[0:2, :], B[-2:, :], B[:, 0:2], B[:, -2:]

    S = np.ravel(S)
    B = np.ravel(B)

    ## calculate ice thickness (surface elevation minus bed elevation)
    H = S - B

    ## volume ice time 1 and time 2
    V_t1 = (1 / constants.sec_yr * dt * H_old * constants.dx * constants.dy * (constants.RHO / 1000.0)).sum()
    V_t2 = (1 / constants.sec_yr * dt * H * constants.dx * constants.dy * (constants.RHO / 1000.0)).sum() + V_t2_edge

    Q_wast = max(0, (V_t1 - V_t2))

    ## ice area in km2
    IA = len(H[H_old > constants.ice_h_min]) * constants.dx * constants.dy / 1e6

    H_IC_jc = 0.5 * (H[ic_jc] + H[im_jc])
    H_ic_JC = 0.5 * (H[ic_jc] + H[ic_jm])

    ## upwind versions of H_IC_jc, H_ic_JC, H_IP_jc, H_ic_JP
    H_IC_jc_up = H[im_jc]
    H_ic_JC_up = H[ic_jm]

    H_IC_jc_up[S[ic_jc] > S[im_jc]] = H[ic_jc[S[ic_jc] > S[im_jc]]]
    H_ic_JC_up[S[ic_jc] > S[ic_jm]] = H[ic_jc[S[ic_jc] > S[ic_jm]]]

    Sx_IC_jc = (S[ic_jc] - S[im_jc]) / constants.dx
    Sy_IC_jc = (S[ic_jp] + S[im_jp] - S[ic_jm] - S[im_jm]) / (4 * constants.dx)
    Sx_ic_JC = (S[ip_jc] + S[ip_jm] - S[im_jc] - S[im_jm]) / (4 * constants.dx)
    Sy_ic_JC = (S[ic_jc] - S[ic_jm]) / constants.dx

    S2_IC_jc = ne.evaluate("Sx_IC_jc**2 + Sy_IC_jc**2")
    S2_ic_JC = ne.evaluate("Sx_ic_JC**2 + Sy_ic_JC**2")

    # no sliding case
    if constants.C_tilde == 0:
        D_IC_jc = ne.evaluate("A_tilde*H_IC_jc_up*H_IC_jc**np1*S2_IC_jc**nm_half")
        D_ic_JC = ne.evaluate("A_tilde*H_ic_JC_up*H_ic_JC**np1*S2_ic_JC**nm_half")

    # sliding casehttps://www.cyberciti.biz/faq/run-execute-sh-shell-script
    elif constants.C_tilde > 0:
        D_IC_jc = ne.evaluate(
            "A_tilde*H_IC_jc_up*H_IC_jc**np1*S2_IC_jc**nm_half + C_tilde*H_IC_jc_up*H_IC_jc**m1*S2_IC_jc**mm_half")
        D_ic_JC = ne.evaluate(
            "A_tilde*H_ic_JC_up*H_ic_JC**np1*S2_ic_JC**nm_half + C_tilde*H_ic_JC_up*H_ic_JC**m1*S2_ic_JC**mm_half")

    else:
        print('C_tilde is undefined or incorrectly defined')

    D_IP_jc = D_IC_jc[ip_jc, 0]
    D_ic_JP = D_ic_JC[ic_jp, 0]
    D_sum = D_IC_jc + D_IP_jc + D_ic_JC + D_ic_JP

    row = np.concatenate((ic_jc, ic_jc, ic_jc, ic_jc, ic_jc))
    col = np.concatenate((im_jc, ip_jc, ic_jm, ic_jp, ic_jc))

    val = np.concatenate((-OMEGA * D_IC_jc, -OMEGA * D_IP_jc,
                          -OMEGA * D_ic_JC, -OMEGA * D_ic_JP,
                          1 / dt + OMEGA * D_sum))

    C = np.concatenate((1 - OMEGA) * (D_IC_jc * S[im_jc] +
                                      D_IP_jc * S[ip_jc] + D_ic_JC * S[ic_jm] +
                                      D_ic_JP * S[ic_jp]) +
                       (1 / dt - (1 - OMEGA) * D_sum) * S[ic_jc] + b_dot)

    A = sparse.coo_matrix((val[:, 0], (row[:, 0], col[:, 0])), shape=(N, N)).tocsc()

    # Flag is whether the conjugate gradient algorithm
    # converged with given interations and tolerances.
    # If it intially fails, try increasing interations
    # to 1000 and tol to 1e-09
    # If that doesn't work try reducing time step

    ##    if flag == 0:
    maxiter = 1000
    tol = 1e-06

    flag = 1

    # M = sparse.coo_matrix((val[:,0],(row[:,0],col[:,0])), shape=(N,N)).tocsc()

    S, flag = cg(A, C, tol=tol, maxiter=maxiter)

    if flag > 0:
        print('Did not converge after ', maxiter, 'times.')

    S = np.maximum(S, B)
    t = t + dt
    H_old = H

    # print(np.round(t, 1), 'in step function')
    # print('Mean ice thickness:', round(np.mean(H[H>ice_h_min]), 2))
    # print('Ice Area    (km^2):',round(IA,2))
    # print('Ice Volume (km^3):', round(1e-06 * IA * np.sum(H[H>ice_h_min])/1000.,2))
    # print('Q_wast:', round(Q_wast, 2))

    return (S, t, H_old, Q_wast, IA, flag)

def LC_update(ice_mask,Class, outClass):
    #glaMask = Path(ice_mask).name
    #Cls = Path(Class).name
    inGlacier = gdal.Open(ice_mask)
    inClass = gdal.Open(Class)
    arrGlacier = inGlacier.GetRasterBand(1).ReadAsArray().astype(np.int)
    arrClass = inClass.GetRasterBand(1).ReadAsArray().astype(np.int)

    for i in range(constants.ny):
        for j in range(constants.nx):
            if arrGlacier[i][j]==0:
                arrClass[i][j]=0
            else:
                arrClass[i][j] = 20

    output = gdal.GetDriverByName('GTiff').Create(outClass,constants.nx,constants.ny,1,gdal.GDT_Int16)
    output.SetGeoTransform(constants.geo)
    output.SetProjection(constants.proj)
    output.GetRasterBand(1).WriteArray(arrClass)
    return outClass

