import os
import math
import librosa
import numpy as np
from pysofaconventions import *

# Reference METU outter trayectory:  bottom outter trayectory
REF_OUT_TRAJ = ["034", "024", "014", "004", "104", "204",
			"304", "404", "504", "604", "614", "624",
			"634", "644", "654", "664", "564", "464",
			"364", "264", "164", "064", "054", "044"]

# Reference METU inner trayectory:  bottom inner trayectory
REF_IN_TRAJ = ["134", "124", "114", "214","314", "414", "514", "524",
                "534", "544", "554", "454", "354", "254", "154", "145"]

def get_mic_xyz():
    """
    Get em32 microphone coordinates in 3D space
    """
    return [(3 - 3) * 0.5, (3 - 3) * 0.5, (2 - 2) * 0.3 + 1.5]

def az_ele_from_source_radians(ref_point, src_point):
    """
    Calculates the azimuth and elevation between a reference point and a source point in 3D space
    Args:
    ref_point (list): A list of three floats representing the x, y, and z coordinates of the reference point
    src_point (list): A list of three floats representing the x, y, and z coordinates of the other point
    Returns:
    A tuple of two floats representing the azimuth and elevation angles in radians plus distance between reference and source point
    """	
    dx = src_point[0] - ref_point[0]
    dy = src_point[1] - ref_point[1]
    dz = src_point[2] - ref_point[2]
    azimuth = math.atan2(dy, dx)
    distance = math.sqrt(dx**2 + dy**2 + dz**2)
    elevation = math.asin(dz/distance)
    return azimuth, elevation, distance

def az_ele_from_source(ref_point, src_point):
    """
    Calculates the azimuth and elevation between a reference point and a source point in 3D space.
    Args:
    ref_point (list): A list of three floats representing the x, y, and z coordinates of the reference point.
    src_point (list): A list of three floats representing the x, y, and z coordinates of the other point.
    Returns:
    A tuple of two floats representing the azimuth and elevation angles in degrees plus the distance between the reference and source points.
    """
    dx = src_point[0] - ref_point[0]
    dy = src_point[1] - ref_point[1]
    dz = src_point[2] - ref_point[2]

    azimuth = math.degrees(math.atan2(dy, dx))
    distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    elevation = math.degrees(math.asin(dz / distance))
    return azimuth, elevation, distance


def compute_azimuth_elevation(receiver_pos, source_pos):
    # Calculate the vector from the receiver to the source
    vector = [source_pos[0] - receiver_pos[0], source_pos[1] - receiver_pos[1], source_pos[2] - receiver_pos[2]]
    # Calculate the azimuth angle
    azimuth = math.atan2(vector[0], vector[1])
    # if azimuth < 0:
    #     azimuth += math.pi
    # Calculate the elevation angle
    distance = math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)
    elevation = math.asin(vector[2] / distance)
    return azimuth, elevation, distance


def get_metu_dataset(aud_fmt="em32"):
    assert aud_fmt == "em32" or aud_fmt == "mic", "You must provide a valid microphone name: em32, mic"
    metu_db_dir = "/scratch/ssd1/RIR_datasets/spargair/em32/"
    top_height = 5
    mic_xyz = get_mic_xyz()
    source_coords, rirs = [], []

    rir_id = 0
    # Outter trayectory: bottom to top
    for height in range(0, top_height):
        for num in REF_OUT_TRAJ:
            # Coords computed based on documentation.pdf from METU Sparg
            x = (3 - int(num[0])) * 0.5
            y = (3 - int(num[1])) * 0.5
            z = (2 - (int(num[2])-height)) * 0.3 + 1.5
            source_xyz = [x, y, z] # note -1 since METU is flipped up-side-down

            azim, elev, _ = az_ele_from_source(mic_xyz, source_xyz)
            elev *= -1 # Account for elevation being swapped in METU

            source_coords.append((rir_id, azim, elev))
            rir_name = num[0] + num[1] + str(int(num[2])-height)
            ir_path = os.path.join(metu_db_dir, rir_name, f"IR_{aud_fmt}.wav")
            irdata, sr = librosa.load(ir_path, mono=False, sr=48000)
            irdata_resamp = librosa.resample(irdata, orig_sr=sr, target_sr=24000)
            irdata_resamp *= 0.3
            rirs.append(irdata_resamp.T)
            rir_id += 1
    return rirs, source_coords


FS = 48000 # original impulse reponse sampling rate
NEW_FS = 24000 # new sampling rate (same as DCASE Synth)

def get_arni_dataset(aud_fmt="em32"):
    assert aud_fmt == "em32" or aud_fmt == "mic", "You must provide a valid microphone name: em32, mic"
    # Load the .sofa file
    room = "arni"
    source_coords, rirs = [], []
    rir_db_path = "/scratch/ssd1/RIR_datasets/6dof_SRIRs_eigenmike_raw/"
    sofa_file_traj = "6DoF_SRIRs_eigenmike_raw_100percent_absorbers_enabled.sofa"

    sofa = SOFAFile(os.path.join(rir_db_path, sofa_file_traj),'r')
    if not sofa.isValid():
        print("Error: the file is invalid")
        return
        
    sourcePositions = sofa.getVariableValue('SourcePosition') # get sound source position
    listenerPosition = sofa.getVariableValue('ListenerPosition') # get mic position
    # get RIR data
    rirdata = sofa.getDataIR()
    num_meas, num_ch = rirdata.shape[0], rirdata.shape[1]
    meas_per_mic = 3 # equal the number of meas per trajectory
    num_meas = 15 # set num_meas to 15 to keep south mics only
    angles_mic_src = [math.degrees(compute_azimuth_elevation(lis, src)[0]) \
            for lis, src in zip(listenerPosition[:num_meas], sourcePositions[:num_meas])]
    meas_sorted_ord = np.argsort(angles_mic_src)[::-1]
    sorted_angles_mic_src = [angles_mic_src[i] for i in meas_sorted_ord]
    doa_xyz, dists, hir_data = [], [], [] # assume only one height
    for rir_id, meas in enumerate(meas_sorted_ord): # for each meas in decreasing order
        # add impulse response
        irdata = rirdata[meas, :, :]
        irdata_resamp = librosa.resample(irdata, orig_sr=FS, target_sr=NEW_FS)
        hir_data.append(irdata_resamp)
        azim, elev, dis = compute_azimuth_elevation(listenerPosition[meas], sourcePositions[meas])
        source_coords.append((rir_id, azim, elev))
        rirs.append(irdata_resamp.T)
        dists.append(dis)
    return rirs, source_coords


def get_audio_spatial_data(room, aud_fmt="em32"):
    if room == "METU":
        rirs, source_coords = get_metu_dataset(aud_fmt)
    elif room == "ARNI":
        rirs, source_coords = get_arni_dataset(aud_fmt)
    else:
        raise ValueError("Unrecognized room name. Please provide either 'METU' or 'ARNI'.")
    return rirs, source_coords 
