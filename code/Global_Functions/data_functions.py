#data_functions.py
# Common GEPA functions to manipulate data
#### Authors: 
# Erin E. McDuffie, Joannes D. Maasakkers, Candice F. Z. Chen
#### Date Last Updated: 
# Feb. 26, 2021

# Import modules
import numpy as np

#Regrid data file from 0.01x0.01 degrees to 0.1x0.1
# inputfile001 = high-res file 
# Lat01/Lon01  = low resolution lat and lon coordiates (select area only)
# output       = low resolution version of inputfile (outfile)
def regrid001_to_01(inputfile001, Lat01, Lon01):
    outfile = np.zeros([len(Lat01),len(Lon01)])
    for ilat in np.arange(len(Lat01)):
        for ilon in np.arange(len(Lon01)):
            outfile[ilat,ilon] = np.sum(inputfile001[ilat*10:(ilat+1)*10,ilon*10:(ilon+1)*10])
    return(outfile)

# Regrid emission flux data from 0.01x0.01 degrees to 0.1x0.1 degrees
# This requires weighting by the grid cell area
# inputfile001 = high-res emission fulx file (molec./cm2/s)
# Lat01/Lon01  = low resolution lat and lon coordiates (select area only)
# area_map001  = high resolution map of grid cell area (m2)
# area_map01   = low resolution map of grid cell area (cm2)
#monthflag     = specify whether file has month resolution (0=no; 1=yes)
# output       = low resolution map of emission fluxes (outfile)
def regrid001_to_01_flux(inputfile001, Lat01, Lon01, area_map001, area_map01, monthflag):
    if monthflag ==0:
        outfile = np.zeros([len(Lat01),len(Lon01)])
    elif monthflag ==1:
        outfile = np.zeros([len(Lat01),len(Lon01), 12])
    for ilat in np.arange(len(Lat01)):
        for ilon in np.arange(len(Lon01)):
            if monthflag ==0:
                outfile[ilat,ilon] = np.sum(inputfile001[ilat*10:(ilat+1)*10,ilon*10:(ilon+1)*10]\
                                      *area_map001[ilat*10:(ilat+1)*10,ilon*10:(ilon+1)*10]*10000)\
                                      /float(area_map01[ilat,ilon]) #to cm^2
            elif monthflag ==1:
                for imonth in np.arange(0,12):
                    outfile[ilat,ilon,imonth] = np.sum(inputfile001[ilat*10:(ilat+1)*10,ilon*10:(ilon+1)*10,imonth]\
                                             *area_map001[ilat*10:(ilat+1)*10,ilon*10:(ilon+1)*10]*10000)\
                                             /float(area_map01[ilat,ilon]) #to cm
    return(outfile)

# Convert meters to degrees
# x, y = locations in meters
def meters2degrees(x, y):
    y = y * 180 / 20037508.34
    lat = np.arctan(np.exp(y * (np.pi / 180))) * 360 / np.pi - 90
    lon = x * 180 / 20037508.34
    return (lat, lon)

#define safe devide to set result to zero if denominator is zero
def safe_div(x,y):
    if y == 0:
        return 0
    return x / y