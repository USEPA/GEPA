#data_plot_functions
# Common GEPA functions to plot data
#### Authors: 
# Erin E. McDuffie, Joannes D. Maasakkers
#### Date Last Updated: 
# Nov. 7, 2023

# Import Modules
import matplotlib.pyplot as plt
import matplotlib.colors as colors
# Set up ticker
import matplotlib.ticker as ticker
from mpl_toolkits.basemap import Basemap
import numpy as np
from copy import copy

# Plot Annual Emission Fluxes (Mg/km/yr)
# Plot one map for each inventory year
# Emi_flux_map = 0.1x0.1 map of emission flux data (molec/cm2/s)
# Lat          = 0.1 degree Lat values (select range)
# Lon          = 0.1 degree Lon values (select range)
# year_range   = array of inventory years
# title_str    = title of map
def plot_annual_emission_flux_map(Emi_flux_map, Lat, Lon, year_range, title_str, scale_max, save_flag, save_outfile):
    # Define constants
    Avogadro   = 6.02214129 * 10**(23)  #molecules/mol
    Molarch4   = 16.04                  #g/mol
    month_day_leap  = [  31,  29,  31,  30,  31,  30,  31,  31,  30,  31,  30,  31]
    month_day_nonleap = [  31,  28,  31,  30,  31,  30,  31,  31,  30,  31,  30,  31]
    
    for iyear in np.arange(len(year_range)): 
        if year_range[iyear]==2012 or year_range[iyear]==2016:
            year_days = np.sum(month_day_leap)
        else:
            year_days = np.sum(month_day_nonleap)
        #my_cmap = copy(plt.cm.get_cmap('rainbow',lut=3000))
        #my_cmap._init()
        #slopen = 200
        #alphas_slope = np.abs(np.linspace(0, 1.0, slopen))
        #alphas_stable = np.ones(3003-slopen)
        #alphas = np.concatenate((alphas_slope, alphas_stable))
        #my_cmap._lut[:,-1] = alphas
        #my_cmap.set_under('gray', alpha=0)
        
        ##Rainbow:
        my_cmap = colors.LinearSegmentedColormap.from_list(name='my_cmap',colors=['#6F4C9B', '#6059A9', '#5568B8', '#4E79C5', '#4D8AC6',
                '#4E96BC', '#549EB3', '#59A5A9', '#60AB9E', '#69B190',
                '#77B77D', '#8CBC68', '#A6BE54', '#BEBC48', '#D1B541',
                '#DDAA3C', '#E49C39', '#E78C35', '#E67932', '#E4632D',
                '#DF4828', '#DA2222', '#B8221E', '#95211B', '#721E17',
                '#521A13'],N=3000)
        my_cmap._init()
        slopen = 200
        alphas_slope = np.abs(np.linspace(0, 1.0, slopen))
        alphas_stable = np.ones(3003-slopen)
        alphas = np.concatenate((alphas_slope, alphas_stable))
        my_cmap._lut[:,-1] = alphas
        my_cmap.set_under('gray', alpha=0)
    
        Lon_cor = Lon[50:632]
        Lat_cor = Lat[43:300]
    
        xpoints = Lon_cor
        ypoints = Lat_cor
        yp,xp = np.meshgrid(ypoints,xpoints)
    
        if np.shape(Emi_flux_map)[0] == len(year_range):
            zp = Emi_flux_map[iyear,43:300,50:632]
        elif np.shape(Emi_flux_map)[2] == len(year_range):
            zp = Emi_flux_map[43:300,50:632,iyear]
        zp = zp/float(10**6 * Avogadro) * (year_days * 24 * 60 * 60) * Molarch4 * float(1e10)
    
        fig, ax = plt.subplots(dpi=300)
        m = Basemap(llcrnrlon=xp.min(), llcrnrlat=yp.min(), urcrnrlon=xp.max(),
                   urcrnrlat=yp.max(), projection='merc', resolution='h', area_thresh=5000)
        m.drawmapboundary(fill_color='Azure')
        m.fillcontinents(color='FloralWhite', lake_color='Azure',zorder=1)
        m.drawcoastlines(linewidth=0.4,zorder=3)
        m.drawstates(linewidth=0.2,zorder=3)
        m.drawcountries(linewidth=0.4,zorder=3)
    
        xpi,ypi = m(xp,yp)
        plot = m.pcolor(xpi,ypi,zp.transpose(), cmap=my_cmap, vmin=10**-15, vmax=scale_max, snap=True,zorder=2,shading='nearest')
        cb = m.colorbar(plot, location = "bottom", pad = "1%")
        tick_locator = ticker.MaxNLocator(nbins=5)
        cb.locator = tick_locator
        cb.update_ticks()
    
        cb.ax.set_xlabel('Methane emissions (Mg a$^{-1}$ km$^{-2}$)',fontsize=10)
        cb.ax.tick_params(labelsize=10)
        Titlestring = str(year_range[iyear])+' '+title_str
        fig1 = plt.gcf()
        plt.title(Titlestring, fontsize=14);
        plt.show();
        if save_flag ==1:
            fig1.savefig(save_outfile+'_'+str(year_range[iyear])+'.tiff',transparent=False)
        
#Plot the difference between the end year and start year (2018-2012)
# Emi_flux_map = 0.1x0.1 map of emission flux data (molec/cm2/s)
# Lat          = 0.1 degree Lat values (select range)
# Lon          = 0.1 degree Lon values (select range)
# year_range   = array of inventory years
def plot_diff_emission_flux_map(Emi_flux_map, Lat, Lon, year_range, title_str,save_flag, save_outfile):
    iyear = len(year_range)-1
    # Define constants
    Avogadro   = 6.02214129 * 10**(23)  #molecules/mol
    Molarch4   = 16.04                  #g/mol
    month_day_leap  = [  31,  29,  31,  30,  31,  30,  31,  31,  30,  31,  30,  31]
    month_day_nonleap = [  31,  28,  31,  30,  31,  30,  31,  31,  30,  31,  30,  31]
    if year_range[iyear]==2012 or year_range[iyear]==2016:
        year_days = np.sum(month_day_leap)
    else:
        year_days = np.sum(month_day_nonleap)
    #my_cmap = copy(plt.cm.get_cmap('RdBu_r',lut=3000))
    #my_cmap._init()
    #slopen = 1501
    #alphas_slope = np.abs(np.linspace(-50, 0, slopen))
    #alphas_stable = np.abs(np.linspace(0, 50, 3003-slopen))
    #alphas = np.concatenate((alphas_slope, alphas_stable))
    #alphas[alphas>1] = 1
    #my_cmap._lut[:,-1] = alphas
    ##Blue-Red without saturation:
    my_cmap = colors.LinearSegmentedColormap.from_list(name='my_cmap',colors=['#2166AC', '#4393C3', '#92C5DE', '#D1E5F0', '#F7F7F7', '#FDDBC7', '#F4A582', '#D6604D', '#B2182B'],N=3000)
    my_cmap._init()
    slopen = 1501
    alphas_slope = np.abs(np.linspace(-50, 0, slopen))
    alphas_stable = np.abs(np.linspace(0, 50, 3003-slopen))
    alphas = np.concatenate((alphas_slope, alphas_stable))
    alphas[alphas>1] = 1
    my_cmap._lut[:,-1] = alphas
    
    Lon_cor = Lon[50:632]
    Lat_cor = Lat[43:300]
    xpoints = Lon_cor
    ypoints = Lat_cor
    yp,xp = np.meshgrid(ypoints,xpoints)
    zp = Emi_flux_map[43:300,50:632,iyear]-Emi_flux_map[43:300,50:632,0]
    zp = zp/float(10**6 * Avogadro) * (year_days * 24 * 60 * 60) * Molarch4 * float(1e10)
    fig, ax = plt.subplots(dpi=300)
    m = Basemap(llcrnrlon=xp.min(), llcrnrlat=yp.min(), urcrnrlon=xp.max(),
               urcrnrlat=yp.max(), projection='merc', resolution='h', area_thresh=5000)
    m.drawmapboundary(fill_color='Azure')
    m.fillcontinents(color='FloralWhite', lake_color='Azure',zorder=1)
    m.drawcoastlines(linewidth=0.4,zorder=3)
    m.drawstates(linewidth=0.2,zorder=3)
    m.drawcountries(linewidth=0.4,zorder=3)
    xpi,ypi = m(xp,yp)
    plot = m.pcolor(xpi,ypi,zp.transpose(), cmap=my_cmap, vmin=-2.5, vmax=2.5, snap=True,zorder=2,shading='nearest')
    cb = m.colorbar(plot, location = "bottom", pad = "1%")
    tick_locator = ticker.MaxNLocator(nbins=5)
    cb.locator = tick_locator
    cb.update_ticks()
    cb.ax.set_xlabel('Methane emissions (Mg a$^{-1}$ km$^{-2}$)',fontsize=10)
    cb.ax.tick_params(labelsize=10)
    Titlestring = title_str
    fig1 = plt.gcf()
    plt.title(Titlestring, fontsize=12);
    plt.show();
    if save_flag ==1:
        fig1.savefig(save_outfile+'_Difference.tiff',transparent=False)
    
    
# Plot Annual Activity Heat Map (Counts/fractions per gridcell per year)
# Plot one map for each inventory year
# Activity_Map = 0.1x0.1 map of activity data (counts or absolute units)
# Plot_Frac    = 0 or 1 (0= plot activity data in absolute counts, 1= plot fractional activity data)
# Lat          = 0.1 degree Lat values (select range)
# Lon          = 0.1 degree Lon values (select range)
# year_range   = array of inventory years
# title_str    = title of map
# legend_str   = title of legend
# scale_max    = maximum of color scale
def plot_annual_activity_map(Activity_Map, Plot_Frac, Lat, Lon, year_range, title_str, legend_str, scale_max):
    # Define constants
    Avogadro   = 6.02214129 * 10**(23)  #molecules/mol
    Molarch4   = 16.04                  #g/mol
    month_day_leap  = [  31,  29,  31,  30,  31,  30,  31,  31,  30,  31,  30,  31]
    month_day_nonleap = [  31,  28,  31,  30,  31,  30,  31,  31,  30,  31,  30,  31]
    
    for iyear in np.arange(len(year_range)): 
        if year_range[iyear]==2012 or year_range[iyear]==2016:
            year_days = np.sum(month_day_leap)
        else:
            year_days = np.sum(month_day_nonleap)
        my_cmap = copy(plt.cm.get_cmap('rainbow',lut=3000))
        my_cmap._init()
        slopen = 200
        alphas_slope = np.abs(np.linspace(0, 1.0, slopen))
        alphas_stable = np.ones(3003-slopen)
        alphas = np.concatenate((alphas_slope, alphas_stable))
        my_cmap._lut[:,-1] = alphas
        my_cmap.set_under('gray', alpha=0)
    
        Lon_cor = Lon[50:632]
        Lat_cor = Lat[43:300]
    
        xpoints = Lon_cor
        ypoints = Lat_cor
        yp,xp = np.meshgrid(ypoints,xpoints)
    
        if np.shape(Activity_Map)[0] == len(year_range):
            if Plot_Frac ==1:
                zp = Activity_Map[iyear,43:300,50:632]/np.sum(Activity_Map[iyear,:,:])
            else:
                zp = Activity_Map[iyear,43:300,50:632]
        elif np.shape(Activity_Map)[2] == len(year_range):
            if Plot_Frac ==1:
                zp = Activity_Map[43:300,50:632,iyear]/np.sum(Activity_Map[:,:,iyear])
            else: 
                zp = Activity_Map[43:300,50:632,iyear]
        #zp = zp/float(10**6 * Avogadro) * (year_days * 24 * 60 * 60) * Molarch4 * float(1e10)
    
        fig, ax = plt.subplots(dpi=300)
        m = Basemap(llcrnrlon=xp.min(), llcrnrlat=yp.min(), urcrnrlon=xp.max(),
                   urcrnrlat=yp.max(), projection='merc', resolution='h', area_thresh=5000)
        m.drawmapboundary(fill_color='Azure')
        m.fillcontinents(color='FloralWhite', lake_color='Azure',zorder=1)
        m.drawcoastlines(linewidth=0.4,zorder=3)
        m.drawstates(linewidth=0.2,zorder=3)
        m.drawcountries(linewidth=0.4,zorder=3)
        
        #if Plot_Frac == 1:
        #    scale_max 
    
        xpi,ypi = m(xp,yp)
        plot = m.pcolor(xpi,ypi,zp.transpose(), cmap=my_cmap, vmin=10**-15, vmax=scale_max, snap=True,zorder=2,shading='nearest')
        cb = m.colorbar(plot, location = "bottom", pad = "1%")
        tick_locator = ticker.MaxNLocator(nbins=5)
        cb.locator = tick_locator
        cb.update_ticks()
    
        cb.ax.set_xlabel(legend_text,fontsize=10)
        cb.ax.tick_params(labelsize=10)
        Titlestring = str(year_range[iyear])+' '+title_str
        plt.title(Titlestring, fontsize=14);
        plt.show();