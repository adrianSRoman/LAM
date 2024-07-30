import os
import sys
import torch
import librosa
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

from trainer.utils import draw_map, get_field, rms_normalizer
from dataset.gen_dataset.gen_dataset import get_visibility_matrix
from dataset.simulated.utils import get_arni_dataset

METU_DATASET = "/scratch/ssd1/RIR_datasets/spargair/em32/"
rirs_list = {"132": 0, "002": -np.pi/4, "032": 0, "112": -np.pi/4} #"362", "262"]
aud_fmt = "em32" # audio format: em32, mic
FS = 24000 # sampling rate
duration = 2 # sythesis duration
num_chans = 32 # get number of channels from RIR
img_indx = 0 # time index image to study (e.g. 10 == 10th ms)
freq_indx = 5 # frequency index to study
bands = np.arange(10)

audio_clips = [
    "/scratch/data/repos/SpatialScaper/datasets/sound_event_datasets/FSD50K_FMA/music/test/Folk/000194.mp3", 
    "/scratch/data/repos/SpatialScaper/datasets/sound_event_datasets/FSD50K_FMA/music/test/Folk/000200.mp3",
    "/scratch/data/repos/SpatialScaper/datasets/sound_event_datasets/FSD50K_FMA/music/test/Hip-Hop/000695.mp3", 
    "/scratch/data/repos/SpatialScaper/datasets/sound_event_datasets/FSD50K_FMA/music/test/Hip-Hop/004682.mp3",
    "/scratch/data/repos/SpatialScaper/datasets/sound_event_datasets/FSD50K_FMA/music/test/International/000666.mp3", 
    "/scratch/data/repos/SpatialScaper/datasets/sound_event_datasets/FSD50K_FMA/music/test/International/000705.mp3",
    "/scratch/data/repos/SpatialScaper/datasets/sound_event_datasets/FSD50K_FMA/music/test/Pop/000822.mp3", 
    "/scratch/data/repos/SpatialScaper/datasets/sound_event_datasets/FSD50K_FMA/music/test/Pop/001661.mp3",
    "/scratch/data/repos/SpatialScaper/datasets/sound_event_datasets/FSD50K_FMA/music/test/Rock/000182.mp3",
    "/scratch/data/repos/SpatialScaper/datasets/sound_event_datasets/FSD50K_FMA/music/test/Rock/000255.mp3"
]

norm_factor = 0
output_sig = np.zeros((10, FS * duration)) # initialize empty output convolved signal
for i, clip in enumerate(audio_clips):
    aud_sig, sr = librosa.load(clip, mono=True, sr=None)
    if sr != FS:
        aud_sig = librosa.resample(aud_sig, orig_sr=sr, target_sr=FS)
    output_sig[i] = aud_sig[:FS * duration]

_,music_norm = rms_normalizer(output_sig)


def distance_simulation(freq_indx, use_noise=False):
    conv_signals = []
    var_dist_list = [] # variance distance list
    max_dist_list = [] # maximum pixel per distance in list
    for rir_tokens in rirs_list:
        clips_var_list = []
        max_azimuth = [] # list of azimuth values of the highest intensity pixel
        for i, clip in enumerate(audio_clips):
            output_sig = np.zeros((FS * duration, num_chans)) # initialize empty output convolved signal
            rir_name = rir_tokens[0] + rir_tokens[1] + rir_tokens[2] # get tokens to RIR names
            ir_path = os.path.join(METU_DATASET, rir_name, f"IR_{aud_fmt}.wav")
            irdata, sr = librosa.load(ir_path, mono=False, sr=FS)
            if sr != FS:
                irdata = librosa.resample(irdata, orig_sr=sr, target_sr=FS, axis=-1)
            #irdata *= 0.3 # Normalize to ~30dBFS
                
            irdata,_ = rms_normalizer(irdata) # appply normalization
            rir_sig = irdata.T        

            aud_sig, sr = librosa.load(clip, mono=True, sr=None)
            if sr != FS:
                aud_sig = librosa.resample(aud_sig, orig_sr=sr, target_sr=FS)
            #aud_sig,_ = rms_normalizer(aud_sig)
            aud_sig /= music_norm[0]

            if use_noise: # overwrite signal with gaussian noise (whitenoise)
                aud_sig = torch.randn(duration*FS)

            # Perform signal convolution - spatialize it
            for ichan in range(num_chans):
                output_sig[:, ichan] += np.convolve(aud_sig[:FS * duration], rir_sig[:FS, ichan], mode='same')
            
            wavfile.write(f'output_{i}.wav', FS, output_sig)
            # get acoustic map
            T_sti = 10e-3 * 20 # 100ms audio frames
            vsg_sig, apgd = get_visibility_matrix(output_sig, FS, apgd=True, bands=bands, T_sti=T_sti) # visibility graph matrix 32ch
            apgd_map = np.abs(apgd[freq_indx, img_indx]) # apdg original shape (nbands, ntime, npixels)
        
            # Get the indices of the top 30 maximum values in apgd_map
            max_indices = np.argpartition(apgd_map, -28)[-28:]
            max_indices = max_indices[np.argsort(-apgd_map[max_indices])]
        
            # get acoustic field and shift to azimuth zero if needed
            R_xyz = get_field(shift_lon=rirs_list[rir_tokens])

            # Get the corresponding coordinates for the top indices
            max_xyz = R_xyz[:, max_indices].T
            max_azimuth.append(max_xyz[0]) # add max element coordinate to the list
            # Calculate the pairwise distances
            dists = np.sqrt(np.sum((max_xyz[0, np.newaxis] - max_xyz[np.newaxis, :])**2, axis=2))
            dists = dists[np.triu_indices_from(dists, k=1)]

            # Calculate the standard deviation of the distances
            std_dist = np.std(dists)
            clips_var_list.append(std_dist**2) # add variance to the list

        var_dist_list.append(clips_var_list) # add list of variances per clips list
        max_dist_list.append(max_azimuth)

    return conv_signals, var_dist_list, max_dist_list


################################################################################
############# Generate plot from the collected data above ######################
################################################################################
for freq_index in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
    conv_signals, var_dist_list, max_dist_list = distance_simulation(freq_index, False)

    gains_labels = [2, 2.8284271247461903, 3, 4.242640687119286]
    mean_variances = [np.mean(np.array(variances)) for variances in var_dist_list]
    std_variances = [np.std(np.array(variances)) for variances in var_dist_list]

    # calculate interval manually using the formula
    y_est = np.array(mean_variances)
    y_err = np.array(std_variances)

    fig, ax = plt.subplots()

    # plot manually calculated interval (std interval) --- the blue one
    ax.plot(np.arange(len(gains_labels)), y_est, '-')
    ax.fill_between(np.arange(len(gains_labels)), y_est - y_err, y_est + y_err, alpha=0.2)

    # Set x-ticks and x-tick labels
    ax.set_xticks(np.arange(len(gains_labels)))
    ax.set_xticklabels(gains_labels)

    # Set x-axis and y-axis labels
    ax.set_title(f'Mean variance across 10 music tracks w.r.t distance \n freq band {freq_index}')
    #ax.set_title(f'Mean variance across of whitenoise source w.r.t distance \n freq band {freq_index}')
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('DoA Intensity Variance (dispersion)')

    fig.savefig(f"./figures/distance_simulation_freq{freq_index}_music.png")
