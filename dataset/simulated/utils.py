import os
import math
import librosa

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


def get_audio_spatial_data(aud_fmt="em32", room="METU"):
    assert aud_fmt == "em32" or aud_fmt == "mic", "You must provide a valid microphone name: em32, mic"

    metu_db_dir = None
    if room == "METU":
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
