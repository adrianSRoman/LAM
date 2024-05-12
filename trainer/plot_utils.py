###### Plotting Utils #######
# Refer to: https://github.com/imagingofthings/DeepWave/blob/master/datasets/Pyramic/color_plot.py
import collections.abc as abc
import scipy.constants as constants
import astropy.coordinates as coord
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import mpl_toolkits.basemap as basemap
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pathlib
import pkg_resources as pkg
import math
import scipy.linalg as linalg
import skimage.util as skutil
from sklearn.cluster import KMeans

eigenmike_raw = {
    "1": [69, 0, 0.042], "2": [90, 32, 0.042], "3": [111, 0, 0.042], "4": [90, 328, 0.042], "5": [32, 0, 0.042],
    "6": [55, 45, 0.042], "7": [90, 69, 0.042], "8": [125, 45, 0.042], "9": [148, 0, 0.042], "10": [125, 315, 0.042],
    "11": [90, 291, 0.042], "12": [55, 315, 0.042], "13": [21, 91, 0.042], "14": [58, 90, 0.042], "15": [121, 90, 0.042],
    "16": [159, 89, 0.042], "17": [69, 180, 0.042], "18": [90, 212, 0.042], "19": [111, 180, 0.042], "20": [90, 148, 0.042],
    "21": [32, 180, 0.042], "22": [55, 225, 0.042], "23": [90, 249, 0.042], "24": [125, 225, 0.042],"25": [148, 180, 0.042],
    "26": [125, 135, 0.042], "27": [90, 111, 0.042], "28": [55, 135, 0.042], "29": [21, 269, 0.042], "30": [58, 270, 0.042],
    "31": [122, 270, 0.042], "32": [159, 271, 0.042],
}

def psnr(label, prediction):
    # maximum possible pixel value
    max_val = np.max(label)
    # Mean Squared Error (MSE)
    mse = np.mean((label - prediction) ** 2)
    # compute peak signal to noise ratio (PSNR)
    psnr = 10 * np.log10(max_val**2 / mse) 
    return psnr

def abs_diff(label, prediction):
    return np.mean(np.abs(label-prediction))

def _deg2rad(coords_dict):
    """
    Take a dictionary with microphone array
    capsules and 3D polar coordinates to
    convert them from degrees to radians
    colatitude, azimuth, and radius (radius
    is left intact)
    """
    return {
        m: [math.radians(c[0]), math.radians(c[1]), c[2]]
        for m, c in coords_dict.items()
    }

def _polar2cart(coords_dict, units=None):
    """
    Take a dictionary with microphone array
    capsules and polar coordinates and convert
    to cartesian
    Parameters:
        units: (str) indicating 'degrees' or 'radians'
    """
    if units == None or units != "degrees" and units != "radians":
        raise ValueError("you must specify units of 'degrees' or 'radians'")
    elif units == "degrees":
        coords_dict = _deg2rad(coords_dict)
    return {
        m: [
            c[2] * math.sin(c[0]) * math.cos(c[1]),
            c[2] * math.sin(c[0]) * math.sin(c[1]),
            c[2] * math.cos(c[0]),
        ]
        for m, c in coords_dict.items()
    }


def get_xyz():
    mic_coords = _polar2cart(eigenmike_raw, units='degrees')
    xyz = [[coord for coord in mic_coords[ch]] for ch in mic_coords]
    return xyz


def eq2cart(r, lat, lon):
    r = np.array([r]) #if chk.is_scalar(r) else np.array(r, copy=False)
    if np.any(r < 0):
        raise ValueError("Parameter[r] must be non-negative.")

    XYZ = (
        coord.SphericalRepresentation(lon * u.rad, lat * u.rad, r)
        .to_cartesian()
        .xyz.to_value(u.dimensionless_unscaled)
    )
    return XYZ

def pol2cart(r, colat, lon):
    lat = (np.pi / 2) - colat
    return eq2cart(r, lat, lon)

def spherical_jn_series_threshold(x, table_lookup=True, epsilon=1e-2):
    if not (0 < epsilon < 1):
        raise ValueError("Parameter[epsilon] must lie in (0, 1).")

    if table_lookup is True:
        rel_path = pathlib.Path("data", "math", "special", "spherical_jn_series_threshold.csv")
        abs_path = pkg.resource_filename("imot_tools", str(rel_path))

        data = pd.read_csv(abs_path).sort_values(by="x")
        x = np.abs(x)
        idx = int(np.digitize(x, bins=data["x"].values))
        if idx == 0:  # Below smallest known x.
            n = data["n_threshold"].iloc[0]
        else:
            if idx == len(data):  # Above largest known x.
                ratio = data["n_threshold"].iloc[-1] / data["x"].iloc[-1]
            else:
                ratio = data["n_threshold"].iloc[idx - 1] / data["x"].iloc[idx - 1]
            n = int(np.ceil(ratio * x))

        return n
    else:

        def series(n, x):
            q = np.arange(n)
            _2q1 = 2 * q + 1
            _sph = special.spherical_jn(q, x) ** 2

            return np.sum(_2q1 * _sph)

        n_opt = int(0.95 * x)
        while True:
            n_opt += 1
            if 1 - series(n_opt, x) < epsilon:
                return n_opt
            
def fibonacci(N, direction=None, FoV=None):            
    if direction is not None:
        direction = np.array(direction, dtype=float)
        direction /= linalg.norm(direction)

        if FoV is not None:
            if not (0 < np.rad2deg(FoV) < 360):
                raise ValueError("Parameter[FoV] must be in (0, 360) degrees.")
        else:
            raise ValueError("Parameter[FoV] must be specified if Parameter[direction] provided.")

    if N < 0:
        raise ValueError("Parameter[N] must be non-negative.")

    N_px = 4 * (N + 1) ** 2
    n = np.arange(N_px)

    colat = np.arccos(1 - (2 * n + 1) / N_px)
    lon = (4 * np.pi * n) / (1 + np.sqrt(5))
    XYZ = np.stack(pol2cart(1, colat, lon), axis=0)

    if direction is not None:  # region-limited case.
        # TODO: highly inefficient to generate the grid this way!
        min_similarity = np.cos(FoV / 2)
        mask = (direction @ XYZ) >= min_similarity
        XYZ = XYZ[:, mask]

    return XYZ


def get_field(min_freq=1500, max_freq=1500,nbands=10):
    freq, bw = (skutil  # Center frequencies to form images
            .view_as_windows(np.linspace(min_freq, max_freq, 10), (2,), 1)
            .mean(axis=-1)), 50.0  # [Hz]

    xyz = get_xyz()
    dev_xyz = np.array(xyz).T
    wl_min = constants.speed_of_sound / (freq.max() + 500)
    sh_order = nyquist_rate(dev_xyz, wl_min) # Maximum order of complex plane waves that can be imaged by the instrument.
    sh_order=10
    R = fibonacci(sh_order)
    R_mask = np.abs(R[2, :]) < np.sin(np.deg2rad(50))
    R = R[:, R_mask]  # Shrink visible view to avoid border effects.
    return R


def nyquist_rate(XYZ, wl):
    baseline = linalg.norm(XYZ[:, np.newaxis, :] - XYZ[:, :, np.newaxis], axis=0)
    N = spherical_jn_series_threshold((2 * np.pi / wl) * baseline.max())
    return N


def wrapped_rad2deg(lat_r, lon_r):
    """
    Equatorial coordinate [rad] -> [deg] unit conversion.
    Output longitude guaranteed to lie in [-180, 180) [deg].
    """
    lat_d = coord.Angle(lat_r * u.rad).to_value(u.deg)
    lon_d = coord.Angle(lon_r * u.rad).wrap_at(180 * u.deg).to_value(u.deg)
    return lat_d, lon_d


def cart2pol(x, y, z):
    """
    Cartesian coordinates to Polar coordinates.
    """
    cart = coord.CartesianRepresentation(x, y, z)
    sph = coord.SphericalRepresentation.from_cartesian(cart)

    r = sph.distance.to_value(u.dimensionless_unscaled)
    colat = u.Quantity(90 * u.deg - sph.lat).to_value(u.rad)
    lon = u.Quantity(sph.lon).to_value(u.rad)

    return r, colat, lon


def cart2eq(x, y, z):
    """
    Cartesian coordinates to Equatorial coordinates.
    """
    r, colat, lon = cart2pol(x, y, z)
    lat = (np.pi / 2) - colat
    return r, lat, lon


def is_scalar(x):
    """
    Return :py:obj:`True` if `x` is a scalar object.
    """
    if not isinstance(x, abc.Container):
        return True

    return False


def eq2cart(r, lat, lon):
    """
    Equatorial coordinates to Cartesian coordinates.
    """
    r = np.array([r]) if is_scalar(r) else np.array(r, copy=False)
    if np.any(r < 0):
        raise ValueError("Parameter[r] must be non-negative.")

    XYZ = (
        coord.SphericalRepresentation(lon * u.rad, lat * u.rad, r)
            .to_cartesian()
            .xyz.to_value(u.dimensionless_unscaled)
    )
    return XYZ


def cmap_from_list(name, colors, N=256, gamma=1.0):
    """
    Parameters
    ----------
    name : str
    colors :
        * a list of (value, color) tuples; or
        * list of color strings
    N : int
        Number of RGB quantization levels.
    gamma : float
        Something?

    Returns
    -------
    cmap : :py:class:`matplotlib.colors.LinearSegmentedColormap`
    """
    from collections.abc import Sized
    import matplotlib.colors

    if not isinstance(colors, abc.Iterable):
        raise ValueError('colors must be iterable')

    if (isinstance(colors[0], Sized) and
            (len(colors[0]) == 2) and
            (not isinstance(colors[0], str))):  # List of value, color pairs
        vals, colors = zip(*colors)
    else:
        vals = np.linspace(0, 1, len(colors))

    cdict = dict(red=[], green=[], blue=[], alpha=[])
    for val, color in zip(vals, colors):
        r, g, b, a = matplotlib.colors.to_rgba(color)
        cdict['red'].append((val, r, r))
        cdict['green'].append((val, g, g))
        cdict['blue'].append((val, b, b))
        cdict['alpha'].append((val, a, a))

    return matplotlib.colors.LinearSegmentedColormap(name, cdict, N, gamma)


def draw_map(I, R, lon_ticks, catalog=None, show_labels=False, show_axis=False, fig=None, ax=None, kmeans=False):
    """
    Parameters
    ==========
    I : :py:class:`~numpy.ndarray`
        (3, N_px)
    R : :py:class:`~numpy.ndarray`
        (3, N_px)
    """

    _, R_el, R_az = cart2eq(*R)
    R_el, R_az = wrapped_rad2deg(R_el, R_az)
    R_el_min, R_el_max = np.around([np.min(R_el), np.max(R_el)])
    R_az_min, R_az_max = np.around([np.min(R_az), np.max(R_az)])

    #fig, ax = plt.subplots()
    bm = basemap.Basemap(projection='mill',
                         llcrnrlat=R_el_min, urcrnrlat=R_el_max,
                         llcrnrlon=R_az_min, urcrnrlon=R_az_max,
                         resolution='c',
                         ax=ax)

    if show_axis:
        bm_labels = [1, 0, 0, 1]
    else:
        bm_labels = [0, 0, 0, 0]
    bm.drawparallels(np.linspace(R_el_min, R_el_max, 5),
                     color='w', dashes=[1, 0], labels=bm_labels, labelstyle='+/-',
                     textcolor='#565656', zorder=0, linewidth=2)
    bm.drawmeridians(lon_ticks,
                     color='w', dashes=[1, 0], labels=bm_labels, labelstyle='+/-',
                     textcolor='#565656', zorder=0, linewidth=2)

    if show_labels:
        ax.set_xlabel('Azimuth (degrees)', labelpad=20)
        ax.set_ylabel('Elevation (degrees)', labelpad=40)

    R_x, R_y = bm(R_az, R_el)
    triangulation = tri.Triangulation(R_x, R_y)

    N_px = I.shape[1]
    mycmap = cmap_from_list('mycmap', I.T, N=N_px)
    colors_cmap = np.arange(N_px)
    ax.tripcolor(triangulation, colors_cmap, cmap=mycmap,
                 shading='gouraud', alpha=0.9, edgecolors='w', linewidth=0.1)

    cluster_center = None
    if kmeans:
        Npts = 6  # find N maximum points
        I_s = np.square(I).sum(axis=0)
        max_idx = I_s.argsort()[-Npts:][::-1]
        x_y = np.column_stack((R_x[max_idx], R_y[max_idx]))  # stack N max energy points
        km_res = KMeans(n_clusters=1).fit(x_y)  # apply k-means to max points
        clusters = km_res.cluster_centers_  # get center of the cluster of N points
        ax.scatter(R_x[max_idx], R_y[max_idx], c='b', s=5)  # plot all N points
        ax.scatter(clusters[:, 0], clusters[:, 1], s=500, alpha=0.3)  # plot the center as a large point
        cluster_center = bm(clusters[:, 0][0], clusters[:, 1][0], inverse=True)

    return fig, ax, cluster_center


def comp_plot(x, y, x_g, y_g, timestamp, azimuth, elevation, ir_times, out_folder, main_title):
    err_az = [a_i - b_i for a_i, b_i in zip(x, x_g)]
    err_el = [a_i - b_i for a_i, b_i in zip(y, y_g)]
    df = {}
    df['azimuth_gt'] = x_g
    df['elevation_gt'] = y_g
    df['azimuth_est'] = x
    df['elevation_est'] = y
    df['azimuth_error'] = err_az
    df['elevation_error'] = err_el
    df['timestamp'] = timestamp
    df = pd.DataFrame(df)

    # plot groundtruth and estimated trajectory
    plt.close("all")
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Trajectory", "Localization Error"))
    fig.add_trace(go.Scatter(x=x, y=y, name='estimated', mode='markers', marker_size=20), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_g, y=y_g, name='ground truth', mode='markers', marker_size=20), row=1, col=1)

    # plot localization error box plot
    fig.add_trace(go.Box(y=df['azimuth_error'].values, name='azimuth error'), row=1, col=2)
    fig.add_trace(go.Box(y=df['elevation_error'].values, name='elevation error'), row=1, col=2)
    fig.update_xaxes(range=[-100, 100], title_text='azimuth', row=1, col=1)
    fig.update_yaxes(range=[-40, 40], title_text='elevation', row=1, col=1)
    fig.update_yaxes(range=[-15, 20], title_text='degree', row=1, col=2)
    fig.update_layout(title_text=main_title, title_x=0.5, title_font_size=40)
    fig.write_html(out_folder + "boxplot.html")
    fig.show()

    # plot azimuth and elevation change over time
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Azimuth over time", "Elevation over time"))
    fig.add_trace(go.Scatter(x=df.timestamp, y=df.azimuth_est, mode='markers',
                             marker_size=abs(df['azimuth_error']) / abs(df['azimuth_error']).max() * 50,
                             name='estimated'), row=1,
                  col=1)
    fig.add_trace(go.Scatter(x=df.timestamp, y=df.azimuth_gt, mode='markers+lines', name='ground truth'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.timestamp, y=df.elevation_est, mode='markers',
                             marker_size=abs(df['elevation_error']) / abs(df['azimuth_error']).max() * 50,
                             name='estimated'),
                  row=1, col=2)
    fig.add_trace(go.Line(x=df.timestamp, y=df.elevation_gt, mode='markers+lines', name='ground truth'), row=1, col=2)
    for n, i in enumerate(ir_times):
        fig.add_vline(x=i * 1000, line_dash='dash', line_color='blue', row=1, col=1)  # at what frame
        fig.add_scatter(x=[i * 1000],
                        y=[azimuth[n]],
                        marker=dict(
                            color='green',
                            size=20
                        ),
                        name='actual gt', row=1, col=1)

        fig.add_vline(x=i * 1000, line_dash='dash', line_color='blue', row=1, col=2)  # at what frame
        fig.add_scatter(x=[i * 1000],
                        y=[elevation[n]],
                        marker=dict(
                            color='green',
                            size=20
                        ),
                        name='actual gt', row=1, col=2)
    fig.update_yaxes(range=[-100, 100], title_text='azimuth', row=1, col=1)
    fig.update_yaxes(range=[-40, 40], title_text='elevation', row=1, col=2)
    fig.update_xaxes(title_text='time', row=1, col=1)
    fig.update_xaxes(title_text='time', row=1, col=2)
    fig.update_layout(title_text=main_title, title_x=0.5, title_font_size=40)
    fig.write_html(out_folder + "time.html")
    fig.show()

