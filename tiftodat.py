import os

def tif_dat(intif, path):
    outasc = path + intif[:-4] + '.asc'
    outdat = path + intif[:-4] + '.dat'
    os.system('gdal_translate -of AAIGrid -ot Float32 ' + intif + ' ' + outasc)
    fasc = open(outasc, 'r')
    header1 = fasc.readline()
    header2 = fasc.readline()
    header3 = fasc.readline()
    header4 = fasc.readline()
    header5 = fasc.readline()
    header6 = fasc.readline()
    fdat = open(outdat, 'w')
    fdat.write(header1)
    fdat.write(header2)
    fdat.write(header3)
    fdat.write(header4)
    fdat.write(header5)
    fdat.write(header6)

    for line in fasc:
        columns = line.split()
        for j in columns:
            fdat.write('%12.4f' % float(j))            
        fdat.write('\n')
    fasc.close()
    fdat.close()
    return outdat





