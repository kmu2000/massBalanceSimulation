# massBalanceSimulation
The provided scripts enable the simulation of daily, monthly, seasonal, and annual glacier mass balance. The modeled mass balance can be calibrated using geodetic mass balance, which is derived from pairs of co-registered DEMs. This process is facilitated by the massBalanceData.py script, which relies on functions from MB_CalcFunc.py.

Climate data for running SnowModel can be sourced from ERA5 Land, accessible through the Copernicus Climate Data Store (https://cds.climate.copernicus.eu/). After creating an account, you can use the ERAL_download_hourly_parallel.py script to download hourly data for a specific location.

To simulate glacier mass balance using the prepared climate data, the scripts Snowmodel_main.py and Snowmodel_parallel.py are employed. Snowmodel_main.py is coupled with an ice dynamics model, accounting for how glacier dynamics redistribute ice and alter the glacier surface, thus affecting mass balance. It uses a constant glacier bed prepared with bed_data_SRTM_Farinottithickness.py, which subtracts the thickness grids from Farinotti et al. (2019) from the SRTM surface. In contrast, Snowmodel_parallel.py simulates mass balance on a constant glacier surface, disregarding ice dynamics. Both scripts utilize functions from Snowmodel_functions.py.


 
