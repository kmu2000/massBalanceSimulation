import rasterio as rasterio
from osgeo import gdal
import numpy as np



# define physical constants for running ice dynamics
# Define physical parameters (here assuming EISMINT-1 values)

n_GLEN = 3.0
m_SLIDE = 0.0
A_GLEN    = 7.5738e-17
C_SLIDE   = 0.0                       # Sliding coefficient in Pa, metre,(Year units)
RHO       = 910.0                     # Density (SI units)
g         = 9.81                      # Gravity (SI units, rho*g has units of Pa)
sec_yr    = 60*60*24*365.15



nm_half = (n_GLEN-1)/2.0
np1     = n_GLEN+1
mm_half = (m_SLIDE-1)/2.0
m1      = m_SLIDE

name = 'RGI15-02999'
# change the path to the bed data
constants_path = '/mnt/e/lidar3/SnowModel/' + name + '/Bed/' 
#glapoly = '/mnt/e/lidar3/SnowModel/SamudraTapu/Extents/RGI_extent.shp'
f_in_bed     = constants_path + 'Bed_' + name + '_srtm.tif'

data         = rasterio.open(f_in_bed)
bed          = data.read(1)

# read the metadata to write output file
profile = data.profile
profile.update(dtype=rasterio.float32, count=1)



## get dimensions of data
nx           = int(data.width)
ny           = int(data.height)
dx = dy      = data.transform[0]


ice_h_min       = 10.0 ## minimum depth to consider pixel is ice and not snow

# subroutine defineconstants

A_tilde = 2.0*A_GLEN*(RHO*g)**n_GLEN/((n_GLEN+2)*dx**2)
C_tilde = C_SLIDE*(RHO*g)**m_SLIDE/dx**2


## open the bed data to read the metadata
inRas = gdal.Open(constants_path + 'Bed_' + name + '_srtm.tif')

## Get top left coordinate and pixel size
geo = inRas.GetGeoTransform()
## Index: [0]top left X, [1]x-pixel size, [2]rotation, [3]top left Y, [4]rotation, [5]y-pixel size

## Get projection information
proj = inRas.GetProjection()

band1 = inRas.GetRasterBand(1).ReadAsArray().astype(np.int)
imagesize = band1.shape


