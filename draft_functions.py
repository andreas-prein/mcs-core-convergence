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

from tqdm import tqdm



def core_2d_properties(Objects,
                      rgiObjectsUD,
                      w_2D,
                      cloudmask,
                      height,
                      lentgh):
    # this function calculates properties of 2D cores
    
    gr_core = {}
    for ob in range(len(Objects)):
        gr_core_act = {}

        w_core = np.copy(w_2D[Objects[ob]])
        w_core[rgiObjectsUD[Objects[ob]] != ob+1] = np.nan
        mask_core = w_core > 0
        
        cloudmask_act = np.copy(cloudmask[Objects[ob]])
        try:
            cloudmask_act[np.isnan(w_core)] = 0
        except:
            stop()
            
        # ignore core if it is partly outside of cloud or touching the boundarie
        if np.sum(~np.isnan(cloudmask_act))/np.sum(np.isnan(cloudmask_act)) < 0.8:
            continue
        
        if Objects[ob] == None:
            continue
        
        heigt_core = height[Objects[ob][0]]
        length_core = (lentgh)[Objects[ob][1]]
        if len(heigt_core) <= 3:
            # cores that are less than four cell deep are not assessed
            continue

        dz_core = heigt_core[1:] - heigt_core[:-1]
        dz_core = np.append(dz_core, dz_core[-1])
        
        dx_core = lentgh[1:] - lentgh[:-1]
        dx_core = np.append(dx_core, dx_core[-1])

        # calculate slope of core
        xx = np.array([np.mean(length_core[mask_core[ll]]) for ll in range(len(heigt_core))])
        core_slope = scipy.stats.linregress(heigt_core,xx).slope

        # plt.pcolormesh(length_core, heigt_core, w_core)
        # plt.scatter(xx,heigt_core, c='k')

        # calculate core width
        core_width = np.array([np.sum(dx_core[Objects[ob][1]][mask_core[ll]] ) for ll in range(len(heigt_core))])
        core_with_mean = np.mean(core_width)
        core_with_max = np.max(core_width)

        # core speed
        core_speed_max = np.nanmax(w_core)
        core_speed_mean = np.nanmean(w_core)

        # core depth
        core_height = np.array([np.sum(dz_core[mask_core[:,ll]]) for ll in range(len(length_core))])
        core_height_mean = np.mean(core_height)
        core_heigth_max = np.max(core_height)

        mean_height = np.mean(np.array([np.mean(heigt_core[mask_core[:,ll]]) for ll in range(len(length_core))]))

        gr_core_act['mean elevation'] = mean_height
        gr_core_act['mean depth'] = core_height_mean
        gr_core_act['max depth'] = core_heigth_max
        gr_core_act['mean speed'] = core_speed_mean
        gr_core_act['max speed'] = core_speed_max
        gr_core_act['mean width'] = core_with_mean
        gr_core_act['max width'] = core_with_max
        gr_core_act['slope from vertical'] = core_slope

        gr_core[str(ob+1)] = gr_core_act

    return gr_core



def core_3d_properties(w_3D,
                      rgiObjectsUD,
                      Objects,
                      height,
                      lat_m,
                      lon_m):

    dx = lon_m[1:] - lon_m[:-1]
    dx = np.append(dx, dx[-1])
    dy = lat_m[1:] - lat_m[:-1]
    dy = np.append(dy, dy[-1])
    dxy = np.meshgrid(dx,dy)[0]

    dx_3d = np.repeat(dxy[np.newaxis, :, :], len(height), axis=0)

    dz = height[1:] - height[:-1]
    dz = np.append(dz, dz[-1])
    dz_3d = np.repeat(dz[:,np.newaxis], len(lat_m), axis=1)
    dz_3d = np.repeat(dz_3d[:,:,np.newaxis], len(lon_m), axis=2)

    gr_core = {}
    for ob in range(len(Objects)):
        gr_core_act = {}
    
        if Objects[ob] == None:
            continue
        heigt_core = Objects[ob][0].stop - Objects[ob][0].start
        
        if heigt_core <= 3:
            # cores that are less than four grid cell deep are not assessed
            continue

        core_ouline_act = rgiObjectsUD[Objects[ob]] == (ob+1)
        # plt.pcolormesh(core_ouline_act[10,:,:])

        core_dx = dx_3d[Objects[ob]] * core_ouline_act
        core_dx[core_dx == 0] = np.nan
        width_x = np.nansum(core_dx, axis=1); width_x[width_x == 0] = np.nan
        width_y = np.nansum(core_dx, axis=2); width_y[width_y == 0] = np.nan
        core_mean_width = np.append(width_x, width_y)
        core_mean_width = core_mean_width[~np.isnan(core_mean_width)]
        core_max_width = np.nanmax([np.nanmax(np.nansum(core_dx, axis=1)) , np.nanmax(np.nansum(core_dx, axis=2))])
        

        area_profile = np.nancumsum(np.nansum(core_dx, axis=(1,2)))
        try:
            center_point = np.where(area_profile/np.nanmax(area_profile) > 0.5)[0][0]
        except:
            continue
        core_mean_elevation = height[Objects[ob][0]][center_point]

        core_dz = dz_3d[Objects[ob]] * core_ouline_act
        core_dz[core_dz == 0] = np.nan
        core_mean_depth = np.nansum(core_dz, axis = 0)
        core_mean_depth = core_mean_depth[core_mean_depth != 0]
        core_max_depth = np.nanmax(np.nansum(core_dz, axis = 0))

        core_w_act = np.copy(w_3D[Objects[ob]]) * core_ouline_act
        core_w_act[core_w_act == 0] = np.nan
        core_w_act[rgiObjectsUD[Objects[ob]] != (ob+1)] = np.nan
        core_mean_speed = np.nanmean(core_w_act)
        core_max_speed = np.nanmax(core_w_act)

        gr_core_act['mean elevation'] = core_mean_elevation
        gr_core_act['mean depth'] = core_mean_depth
        gr_core_act['max depth'] = core_max_depth
        gr_core_act['mean speed'] = core_mean_speed
        gr_core_act['max speed'] = core_max_speed
        gr_core_act['mean width'] = core_mean_width
        gr_core_act['max width'] = core_max_width

        gr_core[str(ob+1)] = gr_core_act
        
    return gr_core




# # Watersheding can be used as an alternative to the breakup function
# # and helps to seperate long-lived/large clusters of objects into sub elements
# def watersheding(label_matrix,  # 2D or 3D matrix with labeled objects [np.array]
#                  WW, # vertical wind field
#                    min_dist,      # minimum distance between two objects [int]
#                    threshold):    # minimum threshold difference between two objects [int]
    
#     import numpy as np
#     from skimage.segmentation import watershed
#     from skimage.feature import peak_local_max
    
#     if len(label_matrix.shape) == 2:
#         conection = np.ones((3, 3))
#     elif len(label_matrix.shape) == 3:
#         conection = np.ones((3, 3, 3))       
    
#     distance = ndimage.distance_transform_edt(label_matrix)
#     local_maxi = peak_local_max(
#                                 WW, 
#                                 footprint=conection, 
#                                 labels=label_matrix, #, indices=False
#                                 min_distance=min_dist, 
#                                 threshold_abs=threshold)
#     peaks_mask = np.zeros_like(distance, dtype=bool)
#     if len(label_matrix.shape) == 2:
#         peaks_mask[local_maxi[:,0], local_maxi[:,1]] = True
#     else:
#         if len(local_maxi[:,0]) != 0:
#             peaks_mask[local_maxi[:,0], local_maxi[:,1], local_maxi[:,2]] = True
        
#     markers = ndimage.label(peaks_mask)[0]
#     labels = watershed(-distance, markers, mask=label_matrix)
    
#     return labels


def watersheding(field_with_max,  # 2D or 3D matrix with labeled objects [np.array]
                   min_dist,      # minimum distance between two objects [int]
                   threshold):    # threshold to identify objects [float]
    
    import numpy as np
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max
    from scipy import ndimage as ndi
    
    if len(field_with_max.shape) == 2:
        conection = np.ones((3, 3))
    elif len(field_with_max.shape) == 3:
        conection = np.ones((3, 3, 3))       

    image =field_with_max > threshold
    coords = peak_local_max(np.array(field_with_max), 
                            min_distance = int(min_dist),
                            threshold_abs = threshold * 1.5,
                            labels = image
                           )
    mask = np.zeros(field_with_max.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(image = np.array(field_with_max)*-1,  # watershedding field with maxima transformed to minima
                       markers = markers, # maximum points in 3D matrix
                       connectivity = conection, # connectivity
                       offset = (np.ones((len(field_with_max.shape))) * 1).astype('int'),
                       mask = image, # binary mask for areas to watershed on
                       compactness = 0) # high values --> more regular shaped watersheds
    
    return labels



def interpolate_obs(w_obs):

    # fill up the small gaps in the dataset
    # https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
    import numpy as np

    def nan_helper(y):
        """Helper to handle indices and logical indices of NaNs.

        Input:
            - y, 1d numpy array with possible NaNs
        Output:
            - nans, logical indices of NaNs
            - index, a function, with signature indices= index(logical_indices),
              to convert logical indices of NaNs to 'equivalent' indices
        Example:
            >>> # linear interpolation of NaNs
            >>> nans, x= nan_helper(y)
            >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
        """

        return np.isnan(y), lambda z: z.nonzero()[0]

    filled_obs = np.copy(w_obs)
    for lev in range(filled_obs.shape[0]):
        try:
            nans, x= nan_helper(w_obs[lev,:])
            filled_obs[lev,:][nans]= np.interp(x(nans), x(~nans), w_obs[lev,:][~nans])
        except:
            continue
            
    return filled_obs


def remove_noise(filled_obs):
    
    filled_obs_clean = np.copy(filled_obs)
    # remove islands of high velocity for up and downdrafts seperately
    n_thresh = 20
    from scipy.ndimage.measurements import label
    for ud in range(2):
        if ud == 0:
            filled_obs_rem = filled_obs > 5
        else:
            filled_obs_rem = filled_obs < 5

        labeled_array, num_features = label(filled_obs_rem)
        binc = np.bincount(labeled_array.ravel())
        noise_idx = np.where(binc >= n_thresh)
        shp = filled_obs_clean.shape
        mask = np.in1d(labeled_array, noise_idx).reshape(shp)
        filled_obs_rem[mask] = 0

        filled_obs_clean[filled_obs_rem] = np.nan

    return filled_obs_clean