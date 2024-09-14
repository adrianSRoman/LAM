import numpy as np
from itertools import combinations
from sklearn.cluster import KMeans
import mpl_toolkits.basemap as basemap

from trainer.utils import cart2eq, wrapped_rad2deg


def determine_similar_location(azi_rad1, lon_rad1, azi_rad2, lon_rad2, thresh_unify=20):
    return distance_between_spherical_coordinates_rad(azi_rad1, lon_rad1, azi_rad2, lon_rad2) < thresh_unify


def distance_between_spherical_coordinates_rad(az1, ele1, az2, ele2):
    """
    The function implemenets the angukla distance between two spherical coordinates
    MORE: https://en.wikipedia.org/wiki/Great-circle_distance
    Parameters :
    ---------------
    :param az1: azimuth angle 1
    :param az2: azimuth angle 2
    :param ele1: elevation angle 1 
    :param ele2: elevation angle 2
    Return:
    ----------------
    :return dist: angular distance in degrees
    """
    dist = np.sin(ele1) * np.sin(ele2) + np.cos(ele1) * np.cos(ele2) * np.cos(np.abs(az1 - az2))
    # Making sure the dist values are in -1 to 1 range, else np.arccos kills the job
    dist = np.clip(dist, -1, 1)
    dist = np.arccos(dist) * 180 / np.pi
    return dist


def get_kmeans_clusters(I, R, catalog=None, N_max=50):
    """
    Parameters
    ==========
    I : :py:class:`~numpy.ndarray`
        (3, N_px)
    R : :py:class:`~numpy.ndarray`
        (3, N_px)
    """
    max_idx = I.argsort()[-N_max:][::-1]
    _, R_el, R_az = cart2eq(*R)
    R_el, R_az = wrapped_rad2deg(R_el, R_az)
    R_el_min, R_el_max = np.around([np.min(R_el), np.max(R_el)])
    R_az_min, R_az_max = np.around([np.min(R_az), np.max(R_az)])
    bm = basemap.Basemap(projection='mill',
                        llcrnrlat=R_el_min, urcrnrlat=R_el_max,
                        llcrnrlon=R_az_min, urcrnrlon=R_az_max)
    R_x, R_y = bm(R_az, R_el)

    weights = I[max_idx] # extract k-means weights

    # Create Kmeans clusters
    K = 3 
    for _k in range(K, 0, -1):
        x_y = np.column_stack((R_x[max_idx], R_y[max_idx]))
        km_res = KMeans(n_clusters=_k).fit(x_y, sample_weight=weights)
        clusters = km_res.cluster_centers_
        centroid_lon, centroid_lat = bm(clusters[:,0], clusters[:,1], inverse=True)

        centroid_lon_rad = centroid_lon * np.pi / 180
        centroid_lat_rad = centroid_lat * np.pi / 180

        all_centroids_pairs = combinations(np.arange(_k), 2)
        centroids_overlap = False
        for _cent_pair in all_centroids_pairs:
            location_overlapping = determine_similar_location(centroid_lon_rad[_cent_pair[0]], centroid_lat_rad[_cent_pair[0]],
                                                              centroid_lon_rad[_cent_pair[1]], centroid_lat_rad[_cent_pair[1]])
            if location_overlapping: # keep looping if overlap between centroids
                centroids_overlap = True
                break
        if not centroids_overlap:   
            break  # done computing K-means centroids

    return centroid_lon, centroid_lat


