#data_IO_functions
# Common GEPA functions to initialize output files
#### Authors: 
# Erin E. McDuffie, Joannes D. Maasakkers, Candice F. Z. Chen
#### Date Last Updated: 
# Feb. 26, 2021

# Load modules
import numpy as np
# Load netCDF (for manipulating netCDF file types)
from netCDF4 import Dataset

# Initiliaze netCDF for the final (0.1x0.1) emission flux data
# outfilename = name of resulting output file
# description = description for the netCDF file meta data
# monthflag   = create file with annual data (=0), create file with monthly data (=1)
# year_range  = array of inventory years
# dimensions  = lat and lon limits
# Lat         = 0.1 degree lat indices
# Lon         = 0.1 degree lon indices
def initialize_netCDF(outfilename, description, monthflag, year_range, dimensions, Lat, Lon):
    #set monthflag to 0 if annual (not monthly) data
    #Initialize file
    nc_out = Dataset(outfilename, 'w', format='NETCDF4')
    nc_out.description = description
    nc_out.year = str(year_range[0])+'-'+str(year_range[-1])

    #Create dimensions
    nc_out.createDimension('year', len(year_range))
    nc_out.createDimension('lat', len(Lat))
    nc_out.createDimension('lon', len(Lon))
    if monthflag ==1:
        nc_out.createDimension('month')

    #Create variables
    year = nc_out.createVariable('year', 'f4', ('year'))
    latitudes = nc_out.createVariable('lat', 'f4', ('lat',))
    longitudes = nc_out.createVariable('lon', 'f4', ('lon',))
    if monthflag ==1:
        month = nc_out.createVariable('month', 'f4', ('month'))
        data_out = nc_out.createVariable('emi_ch4', 'f4', ('lat', 'lon', 'year','month'), zlib=True)
    else:
        data_out = nc_out.createVariable('emi_ch4', 'f4', ('lat', 'lon', 'year'), zlib=True)

    #Properties
    year.standard_name = "year" 
    year.long_name = "Year" 
    year.units = "none"

    longitudes.standard_name = "longitude" 
    longitudes.long_name = "Longitude" 
    longitudes.units = "degrees_east" 

    latitudes.standard_name = "latitude" 
    latitudes.long_name = "Latitude" 
    latitudes.units = "degrees_north" 

    data_out.standard_name = "emissions" 
    data_out.long_name = "Emissions" 
    data_out.units = "moleccm-2s-1"

    #Put location and year data into the arrays
    latitudes[:]  = np.arange(dimensions[0]+0.05,dimensions[1]+0.05,0.1)
    longitudes[:] = np.arange(dimensions[2]+0.05,dimensions[3]+0.05,0.1)
    year[:] = np.arange(year_range[0],year_range[-1]+1) #add one because the range is not inclusive
    if monthflag ==1:
        month[:] = np.arange(1,13)
        
    nc_out.close()
    
    
def initialize_netCDF001(outfilename, description, monthflag, year_range, dimensions, Lat, Lon):
    #set monthflag to 0 if annual (not monthly) data
    #Initialize file
    nc_out = Dataset(outfilename, 'w', format='NETCDF4')
    nc_out.description = description
    nc_out.year = str(year_range[0])+'-'+str(year_range[-1])

    #Create dimensions
    nc_out.createDimension('year', len(year_range))
    nc_out.createDimension('lat', len(Lat))
    nc_out.createDimension('lon', len(Lon))
    if monthflag ==1:
        nc_out.createDimension('month')

    #Create variables
    year = nc_out.createVariable('year', 'f4', ('year'))
    latitudes = nc_out.createVariable('lat', 'f4', ('lat',))
    longitudes = nc_out.createVariable('lon', 'f4', ('lon',))
    if monthflag ==1:
        month = nc_out.createVariable('month', 'f4', ('month'))
        data_out = nc_out.createVariable('emi_ch4', 'f4', ('lat', 'lon', 'year','month'), zlib=True)
    else:
        data_out = nc_out.createVariable('emi_ch4', 'f4', ('lat', 'lon', 'year'), zlib=True)

    #Properties
    year.standard_name = "year" 
    year.long_name = "Year" 
    year.units = "none"

    longitudes.standard_name = "longitude" 
    longitudes.long_name = "Longitude" 
    longitudes.units = "degrees_east" 

    latitudes.standard_name = "latitude" 
    latitudes.long_name = "Latitude" 
    latitudes.units = "degrees_north" 

    data_out.standard_name = "emissions" 
    data_out.long_name = "Emissions" 
    data_out.units = "moleccm-2s-1"

    #Put location and year data into the arrays
    latitudes[:]  = np.arange(dimensions[0]+0.05,dimensions[1]+0.05,0.01)
    longitudes[:] = np.arange(dimensions[2]+0.05,dimensions[3]+0.05,0.01)
    year[:] = np.arange(year_range[0],year_range[-1]+1) #add one because the range is not inclusive
    if monthflag ==1:
        month[:] = np.arange(1,13)
        
    nc_out.close()