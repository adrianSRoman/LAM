import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch import mm

import math
import scipy.linalg as linalg
import skimage.util as skutil
import astropy.coordinates as coord
import astropy.units as u

import scipy.constants as constants

_EIGENMIKE_ = {
    "1": [69, 0, 0.042], "2": [90, 32, 0.042], "3": [111, 0, 0.042], "4": [90, 328, 0.042], "5": [32, 0, 0.042],
    "6": [55, 45, 0.042], "7": [90, 69, 0.042], "8": [125, 45, 0.042], "9": [148, 0, 0.042], "10": [125, 315, 0.042],
    "11": [90, 291, 0.042], "12": [55, 315, 0.042], "13": [21, 91, 0.042], "14": [58, 90, 0.042], "15": [121, 90, 0.042],
    "16": [159, 89, 0.042], "17": [69, 180, 0.042], "18": [90, 212, 0.042], "19": [111, 180, 0.042], "20": [90, 148, 0.042],
    "21": [32, 180, 0.042], "22": [55, 225, 0.042], "23": [90, 249, 0.042], "24": [125, 225, 0.042],"25": [148, 180, 0.042],
    "26": [125, 135, 0.042], "27": [90, 111, 0.042], "28": [55, 135, 0.042], "29": [21, 269, 0.042], "30": [58, 270, 0.042],
    "31": [122, 270, 0.042], "32": [159, 271, 0.042],
}


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
    mic_coords = _polar2cart(_EIGENMIKE_, units='degrees')
    xyz = [[coord for coord in mic_coords[ch]] for ch in mic_coords]
    return xyz


def norm(img, vgg=False):
    if vgg:
        # normalize for pre-trained vgg model
        # https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101
        transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        # normalize [-1, 1]
        transform = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])
    return transform(img)

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

def denorm(img, vgg=False):
    if vgg:
        transform = transforms.Normalize(mean=[-2.118, -2.036, -1.804],
                                         std=[4.367, 4.464, 4.444])
        return transform(img)
    else:
        out = (img + 1) / 2
        return out.clamp(0, 1)
        
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

"""
Special mathematical functions.
"""

import pathlib

import pandas as pd
import pkg_resources as pkg
import scipy.special as special
import pysofaconventions as pysofa

def load_rir_pos(filepath, doas=True):
    sofa = pysofa.SOFAFile(filepath,'r')
    assert sofa.isValid()
    rirs = sofa.getVariableValue('Data.IR')
    source_pos = sofa.getVariableValue('SourcePosition')
    if doas:
        source_pos = source_pos * (1/np.sqrt(np.sum(source_pos**2, axis=1)))[:, np.newaxis] #normalize
    sofa.close()
    return rirs, source_pos


def jv_threshold(x):
    r"""
    Decay threshold of Bessel function :math:`J_{n}(x)`.

    Parameters
    ----------
    x : float

    Returns
    -------
    n : int
        Value of `n` in :math:`J_{n}(x)` past which :math:`J_{n}(x) \approx 0`.
    """
    rel_path = pathlib.Path("data", "math", "special", "jv_threshold.csv")
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


def spherical_jn_threshold(x):
    r"""
    Decay threshold of spherical Bessel function :math:`j_{n}(x)`.

    Parameters
    ----------
    x : float

    Returns
    -------
    n : int
        Value of `n` in :math:`j_{n}(x)` past which :math:`j_{n}(x) \approx 0`.
    """
    rel_path = pathlib.Path("data", "math", "special", "spherical_jn_threshold.csv")
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


def ive_threshold(x):
    r"""
    Decay threshold of the exponentially scaled Bessel function :math:`I_{n}^{e}(x) = I_{n}(x) e^{-|\Re{\{x\}}|}`.

    Parameters
    ----------
    x : float

    Returns
    -------
    n : int
        Value of `n` in :math:`I_{n}^{e}(x)` past which :math:`I_{n}^{e}(x) \approx 0`.
    """
    rel_path = pathlib.Path("data", "math", "special", "ive_threshold.csv")
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


def spherical_jn_series_threshold(x, table_lookup=True, epsilon=1e-2):
    r"""
    Convergence threshold of series :math:`f_{n}(x) = \sum_{q = 0}^{n} (2 q + 1) j_{q}^{2}(x)`.

    Parameters
    ----------
    x : float
    table_lookup : bool
        Use pre-computed table (with `epsilon=1e-2`) to accelerate the search.
    epsilon : float
        Only used when `table_lookup` is :py:obj:`False`.

    Returns
    -------
    n : int
        Value of `n` in :math:`f_{n}(x)` past which :math:`f_{n}(x) \ge 1 - \epsilon`.
    """
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


def jv_series_threshold(x):
    r"""
    Convergence threshold of series :math:`f_{n}(x) = \sum_{q = -n}^{n} J_{q}^{2}(x)`.

    Parameters
    ----------
    x : float

    Returns
    -------
    n : int
        Value of `n` in :math:`f_{n}(x)` past which :math:`f_{n}(x) \ge 1 - \epsilon`.
    """
    rel_path = pathlib.Path("data", "math", "special", "jv_series_threshold.csv")
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

"""
Phased-Array Signal Processing tools.
"""

def get_field(min_freq=1500, max_freq=4500,nbands=10):
    freq, bw = (skutil  # Center frequencies to form images
            .view_as_windows(np.linspace(min_freq, max_freq, 10), (2,), 1)
            .mean(axis=-1)), 50.0  # [Hz]

    xyz = get_xyz()
    dev_xyz = np.array(xyz).T
    wl_min = constants.speed_of_sound / (freq.max() + 500)
    sh_order = nyquist_rate(dev_xyz, wl_min) # Maximum order of complex plane waves that can be imaged by the instrument.
    sh_order=10
    R = fibonacci(sh_order)
    R_mask = np.abs(R[2, :]) < np.sin(np.deg2rad(90))
    R = R[:, R_mask]  # Shrink visible view to avoid border effects.
    return R


def steering_operator(XYZ, R):
    r"""
    Steering matrix.

    Parameters
    ----------
    XYZ : :py:class:`~numpy.ndarray`
        (3, N_antenna) Cartesian array geometry.
    R : :py:class:`~numpy.ndarray`
        (3, N_px) Cartesian grid points in :math:`\mathbb{S}^{2}`.

    Returns
    -------
    A : :py:class:`~numpy.ndarray`
        (N_antenna, N_px) steering matrix.

    Notes
    -----
    The steering matrix is defined as:

    .. math:: {\bf{A}} = \exp \left( -j \frac{2 \pi}{\lambda} {\bf{P}}^{T} {\bf{R}} \right),

    where :math:`{\bf{P}} \in \mathbb{R}^{3 \times N_{\text{antenna}}}` and
    :math:`{\bf{R}} \in \mathbb{R}^{3 \times N_{\text{px}}}`.
    """
    freq, bw = (skutil  # Center frequencies to form images
            .view_as_windows(np.linspace(1500, 4500, 10), (2,), 1)
            .mean(axis=-1)), 50.0  # [Hz]
    wl = constants.speed_of_sound / (freq.max() + 500)
    if wl <= 0:
        raise ValueError("Parameter[wl] must be positive.")

    scale = 2 * np.pi / wl
    A = np.exp((-1j * scale * XYZ.T) @ R)
    return A

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

def nyquist_rate(XYZ, wl):
    """
    Order of imageable complex plane-waves by an instrument.

    Parameters
    ----------
    XYZ : :py:class:`~numpy.ndarray`
        (3, N_antenna) Cartesian array geometry.
    wl : float
        Wavelength [m]

    Returns
    -------
    N : int
        Maximum order of complex plane waves that can be imaged by the instrument.
    """
    baseline = linalg.norm(XYZ[:, np.newaxis, :] - XYZ[:, :, np.newaxis], axis=0)

    N = spherical_jn_series_threshold((2 * np.pi / wl) * baseline.max())
    return N


"""
Linear algebra routines.
"""

import numpy as np
import scipy.linalg as linalg
import scipy.sparse.linalg as splinalg


def eighMax(A):
    r"""
    Evaluate :math:`\mu_{\max}(\bbB)` with

    :math:

    B = (\overline{\bbA} \circ \bbA)^{H} (\overline{\bbA} \circ \bbA)

    Uses a matrix-free formulation of the Lanczos algorithm.

    Parameters
    ----------
    A : :py:class:`~numpy.ndarray`
        (M, N) array.

    Returns
    -------
    D_max : float
        Leading eigenvalue of `B`.
    """
    if A.ndim != 2:
        raise ValueError('Parameter[A] has wrong dimensions.')

    def matvec(v):
        r"""
        Parameters
        ----------
        v : :py:class:`~numpy.ndarray`
            (N,) or (N, 1) array

        Returns
        -------
        w : :py:class:`~numpy.ndarray`
            (N,) array containing :math:`\bbB \bbv`
        """
        v = v.reshape(-1)

        C = (A * v) @ A.conj().T
        D = C @ A
        w = np.sum(A.conj() * D, axis=0).real
        return w

    M, N = A.shape
    B = splinalg.LinearOperator(shape=(N, N),
                                matvec=matvec,
                                dtype=np.float64)
    D_max = splinalg.eigsh(B, k=1, which='LM', return_eigenvectors=False)
    return D_max[0]


def psf_exp(XYZ, R, wl, center):
    """
    True complex plane-wave point-spread function.

    Parameters
    ----------
    XYZ : :py:class:`~numpy.ndarray`
        (3, N_antenna) Cartesian instrument coordinates.
    R : :py:class:`~numpy.ndarray`
        (3, N_px) Cartesian grid points.
    wl : float
        Wavelength of observations [m].
    center : :py:class:`~numpy.ndarray`
        (3,) Cartesian position of PSF focal point.

    Returns
    -------
    psf_mag2 : :py:class:`~numpy.ndarray`
        (N_px,) PSF squared magnitude.
    """
    N_antenna = XYZ.shape[1]
    if not (XYZ.shape == (3, N_antenna)):
        raise ValueError('Parameter[XYZ] must be (3, N_antenna) real-valued.')

    N_px = R.shape[1]
    if not (R.shape == (3, N_px)):
        raise ValueError('Parameter[R] must be (3, N_px) real-valued.')

    if not (wl > 0):
        raise ValueError('Parameter[wl] must be positive.')

    if not (center.shape == (3,)):
        raise ValueError('Parameter[center] must be (3,) real-valued.')

    A = phased_array.steering_operator(XYZ, R, wl)
    d = phased_array.steering_operator(XYZ, center.reshape(3, 1), wl)

    psf = np.reshape(d.T.conj() @ A, (N_px,))
    psf_mag2 = np.abs(psf) ** 2
    return psf_mag2


def psf_sinc(XYZ, R, wl, center):
    """
    Asymptotic point-spread function for uniform spherical arrays as antenna
    density converges to 1.

    Parameters
    ----------
    XYZ : :py:class:`~numpy.ndarray`
        (3, N_antenna) Cartesian instrument coordinates.
    R : :py:class:`~numpy.ndarray`
        (3, N_px) Cartesian grid points.
    wl : float
        Wavelength of observations [m].
    center : :py:class:`~numpy.ndarray`
        (3,) Cartesian position of PSF focal point.

    Returns
    -------
    psf_mag2 : :py:class:`~numpy.ndarray`
        (N_px,) PSF squared magnitude.
    """
    N_antenna = XYZ.shape[1]
    if not (XYZ.shape == (3, N_antenna)):
        raise ValueError('Parameter[XYZ] must be (3, N_antenna) real-valued.')

    N_px = R.shape[1]
    if not (R.shape == (3, N_px)):
        raise ValueError('Parameter[R] must be (3, N_px) real-valued.')

    if not (wl > 0):
        raise ValueError('Parameter[wl] must be positive.')

    if not (center.shape == (3,)):
        raise ValueError('Parameter[center] must be (3,) real-valued.')

    XYZ_centroid = np.mean(XYZ, axis=1, keepdims=True)
    XYZ_radius = np.mean(linalg.norm(XYZ - XYZ_centroid, axis=0))
    center = center / linalg.norm(center)

    psf = np.sinc((2 * XYZ_radius / wl) *
                  linalg.norm(R - center.reshape(3, 1), axis=0))
    psf_mag2 = psf ** 2
    return psf_mag2
