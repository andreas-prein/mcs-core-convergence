#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python


# # 4km_comgrid_comparison.ipynb

# In[2]:


'''
This program reads in data from the MCS simulations on the common grid
and analyzes if there are systematic differences and convergence
'''


# In[3]:


from dateutil import rrule
import datetime
import glob
from netCDF4 import Dataset
import sys, traceback
import dateutil.parser as dparser
import string
from pdb import set_trace as stop
import numpy as np
import numpy.ma as ma
import os
# from mpl_toolkits import basemap
# import ESMF
import pickle
import subprocess
import pandas as pd
from scipy import stats
import copy
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib as mpl
import pylab as plt
import random
import scipy.ndimage as ndimage
import scipy
import shapefile
import matplotlib.path as mplPath
from matplotlib.patches import Polygon as Polygon2
# Cluster specific modules
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.vq import kmeans2,vq, whiten
from scipy.ndimage import gaussian_filter
# import seaborn as sns
# import metpy.calc as mpcalc
import shapefile as shp
import sys 
import matplotlib.gridspec as gridspec
import seaborn
# from mpl_toolkits.basemap import Basemap, cm
from wrf import (to_np, getvar, smooth2d, get_cartopy, cartopy_xlim,
                 cartopy_ylim, latlon_coords)
import cartopy.crs as ccrs

from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import wrf

import metpy
from metpy.calc import density
from metpy.units import units


# In[6]:


from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from metpy.plots import ctables
# READ AND PARSE HERE
dbzmap = ctables.registry.get_colortable('NWSReflectivity')
# dbzmap = ctables.registry.get_colortable('NWSStormClearReflectivity')


# In[7]:


def disctance(lat1, lon1, lat2, lon2):
    from math import sin, cos, sqrt, atan2, radians

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    
    # Approximate radius of earth in km
    R = 6373.0

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c
    
    return distance


# In[8]:


def variogram_drafts(ww,
                    threshold,
                    lag_dist,
                    draft_num):

    from draft_functions import core_2d_properties, core_3d_properties, watersheding
    rgiObj_Struct=np.zeros((3,3,3)); rgiObj_Struct[:,:,:]=1
    lag_dist_c = (lag_dist[1:] + lag_dist[:-1]) / 2
    variogram = np.zeros((draft_num, len(lag_dist_c))); variogram[:] = np.nan
    
    if np.max(ww) >= threshold:
        
        ww_th = ww > threshold
        rgiObjectsUD, nr_objectsUD = ndimage.label(ww_th,structure=rgiObj_Struct[0,:,:])
        rgiObjectsUD = watersheding(rgiObjectsUD, 
                                    ww,
                                     1,
                                     1)
        
        # find the maximum in drafts and select the top 20 drafts for further analysis
        
        Objects=ndimage.find_objects(rgiObjectsUD)
        draft_max = np.zeros((len(Objects))); draft_max[:] = np.nan
        for ii in range(len(Objects)):
            draft_act = np.copy(ww[Objects[ii]])
            draft_max[ii] = np.max(draft_act[rgiObjectsUD[Objects[ii]] == ii+1])
        
        draft_id = np.argsort(draft_max)[::-1][:draft_num]
        
        
        for dr in range(np.min([draft_num, len(draft_id)])):
            # find location of draft maximum
            draft_act = np.copy(ww)
            draft_act[rgiObjectsUD != draft_id[dr]+1] = 0
            max_loc = np.where(draft_act == np.max(draft_act))
            max_val = draft_max[draft_id[dr]]
            
            dist = disctance(lat[max_loc[0][0], max_loc[1][0]],
                    lon[max_loc[0][0], max_loc[1][0]],
                    lat,
                    lon)
            for dd in range(len(lag_dist_c)):
                points = (dist >= lag_dist[dd]) & (dist < lag_dist[dd+1])
                variogram[dr,dd] = np.var(ww[points] - max_val)
    return variogram


# In[9]:


# from draft_functions import core_2d_properties, core_3d_properties, \
#                             watersheding, interpolate_obs, remove_noise


# In[10]:


# DX = ['500M','250M','125M']
# DT = [2,1,0.5]
# dx_m = [500,250,125] 

DX_all = ['4KM','2KM','1KM','500M','250M','125M'] #,'obs']
colors = ['#a6cee3','#33a02c','#ff7f00','#fb9a99','#e31a1c']
color_cortype = ['#e41a1c', '#377eb8', '#33a02c']
dx_km = [4,2,1,0.5,0.25,0.125]
DT_all = [16,8,4,2,1,0.5]
dx_m_all = [4000,2000,1000,500,250,125]

# for variogram calculatoin
lag_dist = np.arange(0,104,4)
draft_num = 20
threshold = 1.5

sim = int(sys.argv[1])

SIM_All = ['mao_20140401_15:00:00_', #125 | 0
        'mao_20140917_17:00:00_',
        'mao_20141004_13:00:00_',
        'mao_20141018_14:00:00_',
        'mao_20141117_18:00:00_',
        'mao_20141210_14:00:00_',
        'mao_20150328_15:00:00_',
        'mao_20150412_12:00:00_',
        'mao_20150621_14:00:00_',
        'mao_20151106_12:00:00_', #125 | 9
        'sgp_20120531_04:00:00_',
        'sgp_20120615_07:00:00_', #125 | 11
        'sgp_20130509_07:00:00_',
        'sgp_20130605_09:00:00_',
        'sgp_20130617_07:00:00_',
        'sgp_20140602_04:00:00_',
        'sgp_20140605_12:00:00_',
        'sgp_20140612_06:00:00_', #125 | 17
        'sgp_20140628_16:00:00_',
        'sgp_20140710_10:00:00_']
    
wrfout_dir = '/glade/campaign/mmm/c3we/prein/Projects/2019_ASR-MCS/data/Coarsened_Data/4km_more-vars/'
save_dir = '/glade/campaign/mmm/c3we/prein/Projects/2019_ASR-MCS/data/Coarsened_Data/processed_data/'


# In[11]:


# if Site == 'SGP':
#     proj_File = '/glade/campaign/mmm/c3we/mingge/WRFV4.1.5_intel_dmpar/Thomson_YSU/sgp_20120531_04:00:00_L4/wrfout_d01_2012-05-30_04:00:00'
#     extension = [-111, -88, 28, 48]
#     ARMlon = -97.4882
#     ARMlat = 36.6077
# if Site == 'MAO':
#     proj_File = '/glade/campaign/mmm/c3we/mingge/WRFV4.1.5_intel_dmpar/Thomson_YSU/mao_20150303_20:00:00_L4/wrfout_d01_2015-03-02_20:00:00'
#     extension = [-69, -49, -13.5,6]
#     ARMlon = -60.025
#     ARMlat = -3.113


# ### Read in W and Z at time t_foc

# In[12]:


# define matrices to store the processed data
# diabatic_heating_all = np.zeros((72,95, len(SIM_All), len(DX_all))); diabatic_heating_all[:] = np.nan
# massflux_up_all = np.copy(diabatic_heating_all)
# massflux_down_all = np.copy(diabatic_heating_all)
# q_frozen_all = np.copy(diabatic_heating_all)
# q_graupel_all = np.copy(diabatic_heating_all)
# q_liquid_all = np.copy(diabatic_heating_all)
# thetae_all = np.copy(diabatic_heating_all)

variables = ['diabtic heating', 
             'upward mass flux',
            'downward mass flux',
            'frozen mixing ratio',
            'grapel mixing ratio',
            'liquid mixing ratio',
            'Theta E']
areal_av = np.zeros((72,95, len(SIM_All), len(DX_all), len(variables))); areal_av[:] = np.nan

pr_all = np.zeros((72,187,187, len(SIM_All), len(DX_all))); pr_all[:] = np.nan

variograms = np.zeros((72, 95, draft_num, len(lag_dist)-1, len(SIM_All), len(DX_all), 2)); variograms[:] = np.nan


# In[ ]:


for si in [sim]: #range(len(SIM_All)):
    save_sim = save_dir+SIM_All[si]+'_4km-comgrid_analysis.npz'
    if os.path.isfile(save_sim+'*') == False:
        areal_av_sim = np.copy(areal_av[:,:,si,:])
        pr_sim = np.copy(pr_all[:,:,:,si,:])
        variograms_sim = np.copy(variograms[:,:,:,:,si,:])
        SIM = SIM_All[si]
        print(SIM)
    
        if SIM[:3] == 'sgp':
            #  location of SGP site
            ARMlon = -97.4882
            ARMlat = 36.6077
            mean_height = np.array([   69.,   131.,   210.,   310.,   436.,   593.,   785.,  1020.,
                            1294.,  1583.,  1870.,  2154.,  2435.,  2714.,  2991.,  3266.,
                            3538.,  3809.,  4078.,  4345.,  4610.,  4874.,  5137.,  5398.,
                            5658.,  5916.,  6173.,  6429.,  6684.,  6938.,  7190.,  7441.,
                            7692.,  7941.,  8189.,  8436.,  8681.,  8926.,  9170.,  9412.,
                            9654.,  9894., 10134., 10372., 10609., 10846., 11081., 11315.,
                            11547., 11778., 12007., 12235., 12460., 12684., 12906., 13127.,
                            13346., 13563., 13779., 13993., 14208., 14422., 14636., 14851.,
                            15065., 15280., 15494., 15709., 15923., 16137., 16352., 16566.,
                            16780., 16994., 17208., 17422., 17636., 17850., 18064., 18277.,
                            18491., 18705., 18918., 19132., 19346., 19560., 19773., 19987.,
                            20201., 20415., 20629., 20843., 21057., 21271., 21485.])
        elif SIM[:3] == 'mao':
            #  location of Mao site
            ARMlon = -60.025
            ARMlat = -3.113 
            mean_height = np.array([   37.,    99.,   180.,   281.,   409.,   569.,   766.,  1007.,
                            1290.,  1589.,  1887.,  2183.,  2476.,  2768.,  3058.,  3346.,
                            3633.,  3917.,  4200.,  4481.,  4760.,  5038.,  5314.,  5588.,
                            5860.,  6131.,  6400.,  6668.,  6934.,  7198.,  7461.,  7722.,
                            7981.,  8239.,  8495.,  8750.,  9003.,  9254.,  9504.,  9752.,
                            9998., 10243., 10486., 10727., 10967., 11205., 11442., 11677.,
                            11910., 12141., 12371., 12598., 12824., 13049., 13271., 13491.,
                            13710., 13927., 14143., 14357., 14570., 14784., 14997., 15210.,
                            15423., 15636., 15849., 16061., 16274., 16487., 16699., 16912.,
                            17124., 17337., 17549., 17762., 17975., 18187., 18400., 18613.,
                            18826., 19039., 19252., 19465., 19678., 19891., 20104., 20317.,
                            20530., 20743., 20957., 21170., 21383., 21597., 21810.])
        
        for dx in tqdm(range(len(DX_all))):
            print(DX_all[dx])
        
            if np.isin(DX_all[dx], ('500M','250M','125M')) == True:
                subkm = True
            else:
                subkm = False
        
            if subkm == True:
                wrfout_files = np.sort(glob.glob(wrfout_dir+DX_all[dx]+'/'+SIM+'L/wrfout_d02*'))
            else:
                if DX_all[dx] != '12KM':
                    wrfout_files = np.sort(glob.glob(wrfout_dir+DX_all[dx]+'/'+SIM+'L/wrfout_d01*'))
                else:
                    wrfout_files = np.sort(glob.glob(wrfout_dir+DX_all[dx]+'/'+SIM+'L/wrfout_d01*'))
                # only focus the analysis on 18 - 30 hours 
                wrfout_files = wrfout_files[18*6+1:18*6+1+12*6]
    
            for hh in range(len(wrfout_files)):
                # read in key variables at t_foc
                ncfile = Dataset(wrfout_files[hh])
                HGT = np.array(getvar(ncfile, "HGT"))
                lat = np.squeeze(ncfile.variables["XLAT"])
                lon = np.squeeze(ncfile.variables['XLONG'])
                dbz = np.squeeze(ncfile.variables["REFL_10CM"][:,:,:,:])
                ww = np.squeeze(ncfile.variables["W"][:,:,:,:])
                zz = (np.squeeze(ncfile.variables["PHB"][:,:,:,:]) + np.squeeze(ncfile.variables["PH"][:,:,:,:]))/9.81 - HGT[None,:]
                pp = np.squeeze(ncfile.variables["P"][:,:,:,:]) + np.squeeze(ncfile.variables["PB"][:,:,:,:])
                tt = np.squeeze(ncfile.variables["T"][:,:,:,:])
                qv = np.squeeze(ncfile.variables["QVAPOR"][:,:,:,:])
                qi = np.squeeze(ncfile.variables["QICE"][:,:,:,:])
                qg = np.squeeze(ncfile.variables["QGRAUP"][:,:,:,:])
                qr = np.squeeze(ncfile.variables["QRAIN"][:,:,:,:])
                qs = np.squeeze(ncfile.variables["QSNOW"][:,:,:,:])
                qc = np.squeeze(ncfile.variables["QCLOUD"][:,:,:,:])
                if 'H_DIABATIC' in ncfile.variables.keys():
                    diab_heat = np.squeeze(ncfile.variables["H_DIABATIC"][:,:,:,:])
                    RTHRATSW = np.squeeze(ncfile.variables["RTHRATSW"][:,:,:,:])
                    RTHRATLW = np.squeeze(ncfile.variables["RTHRATLW"][:,:,:,:])
                if SIM[:3] == 'sgp':
                    pr_sim[hh,:,:,dx] = np.squeeze(ncfile.variables["RAINNC"][:,:-1,:-1])
                else:
                    pr_sim[hh,:,:,dx] = np.squeeze(ncfile.variables["RAINNC"][:,:,:])
                ncfile.close()
    
                tk = wrf.tk(pp, tt+300, meta=False, units='K')
                ww_cent = (ww[:-1,:] + ww[1:,:]) / 2
                zz_cent = (zz[:-1,:] + zz[1:,:]) / 2
                
                # calculate air density
                pp = units.Quantity(pp, "Pa")
                tk = units.Quantity(tk, "degK")
                qtot = units.Quantity(((qv+qi+qs+qg+qc+qr) * 1000), 'g/kg')
                
                dens_moist = density(pp, 
                                   tk, 
                                   qtot)
    
                # calculate theta-e
                qv = units.Quantity((qv * 1000), 'g/kg')
                p_vap = metpy.calc.vapor_pressure(pp, qv)
                dewpoint = metpy.calc.dewpoint(p_vap)
                thetae = metpy.calc.equivalent_potential_temperature(pp, tk, dewpoint)

                # calculate variogram of vertical wind speed

                # Vertical mass transport
                ww_up = np.copy(ww_cent)
                ww_up[ww_up < 0] = 0
                vert_mass_up = dens_moist * ww_up * 4000**2 
                ww_down = np.copy(ww_cent)
                ww_down[ww_down > 0] = 0
                vert_mass_down = dens_moist * ww_down * 4000**2
    
                # interpolate to common height grid
                vert_mass_up_int = np.array(wrf.interpz3d(np.array(vert_mass_up), 
                                                 np.array(zz_cent), 
                                                 np.array(mean_height)))
                vert_mass_up_int[vert_mass_up_int > 10**10] = np.nan
                areal_av_sim[hh,:,dx,variables.index('upward mass flux')] = np.mean(vert_mass_up_int, axis=(1,2))
                
                vert_mass_down_int = np.array(wrf.interpz3d(np.array(vert_mass_down), 
                                                 np.array(zz_cent), 
                                                 np.array(mean_height)))
                vert_mass_down_int[vert_mass_down_int > 10**10] = np.nan
                areal_av_sim[hh,:,dx,variables.index('downward mass flux')] = np.mean(vert_mass_down_int, axis=(1,2))
    
                if 'H_DIABATIC' in ncfile.variables.keys():
                    diab_heat_int = np.array(wrf.interpz3d(np.array(diab_heat), 
                                                     np.array(zz_cent), 
                                                     np.array(mean_height)))
                    diab_heat_int[diab_heat_int > 10**10] = np.nan
                    areal_av_sim[hh,:,dx,variables.index('diabtic heating')] = np.nanmean(diab_heat_int, axis=(1,2))
    
                qgraup_int = np.array(wrf.interpz3d(np.array(qg), 
                                                 np.array(zz_cent), 
                                                 np.array(mean_height)))
                qgraup_int[qgraup_int > 10**10] = np.nan
                areal_av_sim[hh,:,dx,variables.index('grapel mixing ratio')] = np.mean(qgraup_int, axis=(1,2))
    
                qfrozen_int = np.array(wrf.interpz3d(np.array((qg+qi+qs)), 
                                                 np.array(zz_cent), 
                                                 np.array(mean_height)))
                qfrozen_int[qfrozen_int > 10**10] = np.nan
                areal_av_sim[hh,:,dx,variables.index('frozen mixing ratio')] = np.mean(qfrozen_int, axis=(1,2))
    
                qliquid_int = np.array(wrf.interpz3d(np.array((qc+qr)), 
                                                 np.array(zz_cent), 
                                                 np.array(mean_height)))
                qliquid_int[qliquid_int > 10**10] = np.nan
                areal_av_sim[hh,:,dx,variables.index('liquid mixing ratio')] = np.mean(qliquid_int, axis=(1,2))
    
                thetae_int = np.array(wrf.interpz3d(np.array(thetae), 
                                                 np.array(zz_cent), 
                                                 np.array(mean_height)))
                thetae_int[thetae_int > 10**10] = np.nan
                areal_av_sim[hh,:,dx,variables.index('Theta E')] = np.mean(thetae_int, axis=(1,2))

                # calculate variogram
                ww_int = np.array(wrf.interpz3d(np.array(ww_cent), 
                                                 np.array(zz_cent), 
                                                 np.array(mean_height)))
                ww_int[ww_int > 10**10] = np.nan
                for lev in range(ww_int.shape[0]):
                    # updrafts
                    variograms_sim[hh, lev,:,:,dx,0] = \
                    variogram_drafts(ww_int[lev,:,:],
                                            threshold,
                                            lag_dist,
                                            draft_num)
                    # downdrafts
                    variograms_sim[hh, lev,:,:,dx,1] = \
                    variogram_drafts(ww_int[lev,:,:]*-1,
                                            threshold,
                                            lag_dist,
                                            draft_num)
        
        np.savez(save_sim,
                areal_av_sim = areal_av_sim,
                pr_sim = pr_sim,
                variograms_sim = variograms_sim)
    else:
        data = np.load(save_sim)
        areal_av_sim = data['areal_av_sim']
        pr_sim = data['pr_sim']
        variograms_sim = data['variograms_sim']

    areal_av[:,:,si,:] = areal_av_sim
    pr_all[:,:,:,si,:] = pr_sim
    variograms[:,:,:,:,si,:] = variograms_sim







