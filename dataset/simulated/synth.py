import os
import yaml
import copy
import random
import librosa
import argparse
import soundfile as sf
import numpy as np
from tqdm import tqdm

import scipy
import scipy.signal as signal

from utils import *

DCASE_EVENTS = "/scratch/data/repos/SpatialScaper/datasets/sound_event_datasets/FSD50K_FMA"

DCASE_SOUND_EVENT_CLASSES = {
    "femaleSpeech": 0,
    "maleSpeech": 1,
    "clapping": 2,
    "telephone": 3,
    "laughter": 4,
    "domesticSounds": 5,
    "footsteps": 6,
    "doorCupboard": 7,
    "music": 8,
    "musicInstrument": 9,
    "waterTap": 10,
    "bell": 11,
    "knock": 12,
}

LEAVE_OUT_CLASSES = ["femaleSpeech", "maleSpeech", "clapping", "telephone", "laughter", "domesticSounds", "footsteps",
                    "doorCupboard", "musicInstrument",  "waterTap", "bell", "knock"]


def get_audio_tracks(dataset_path=DCASE_EVENTS, leave_out_classes=LEAVE_OUT_CLASSES, split="train"):
    tracks_list = []
    for event_class in DCASE_SOUND_EVENT_CLASSES.keys():
        if event_class in leave_out_classes:
            continue
        
        for dirpath, dirnames, tracks in os.walk(os.path.join(dataset_path, event_class, split)):
            # ensure no subdirectories
            if len(dirnames) == 0:
                for track in tracks:
                    tracks_list.append((os.path.join(dirpath, track), DCASE_SOUND_EVENT_CLASSES[event_class]))        
    return tracks_list


class AudioSynthesizer:
    def __init__(self, rirs, source_coords, audio_tracks_paths, total_duration):
        self.rirs = rirs # room_impulse responses
        self.source_coords = source_coords
        self.audio_tracks_paths = audio_tracks_paths
        self.total_duration = total_duration
        self.num_chans = self.rirs[0].shape[1] # channel count in array
        self.audio_FS = 24000	# sampling rate (24kHz)
        self.total_frames = int(self.total_duration * self.audio_FS)  # Calculate total frames based on total_duration


    def spatialize_audio_events(self, n_polyphony=1, track_num=0, dest_dir="./output"):
        # generate a copy of the available rirs to be used
        rirs_list, coords_list = copy.deepcopy(self.rirs), copy.deepcopy(self.source_coords)
        # intitalize convolve audio track
        conv_sig = np.zeros((self.total_frames, self.num_chans))
        # Define output filename: <track_num>_<polyphony>_<class_id_1>_<azi_1>_<ele_1>_<class_id_2>_<azi_2>_<ele_2>_... .wav
        output_filename = f"{track_num+1:03d}_polyphony{n_polyphony}"
        for _ in range(n_polyphony):
            # get random room impulse reponse
            rir_idx = random.randrange(len(rirs_list))
            rir_sig, coord = rirs_list.pop(rir_idx), coords_list.pop(rir_idx) 
            # get random sound event to spatialize
            event_idx = random.randrange(len(self.audio_tracks_paths))
            # load audio and resample if neccesary
            event, class_id = self.audio_tracks_paths[event_idx]
            event_sig, sr = librosa.load(event, sr=None, mono=None)
            if sr != self.audio_FS:
                event_sig = librosa.resample(event_sig, orig_sr=sr, target_sr=self.audio_FS)
            event_sig = event_sig.reshape(-1, 1)
            event_sig = np.tile(event_sig, (1, rir_sig.shape[-1]))
            # place event within the pre-defined duration
            if event_sig.shape[0] < self.total_frames:
                temp_event = np.zeros((self.total_frames, self.num_chans))
                start_idx = random.randint(0, self.total_frames-event_sig.shape[0]) # choose a starting index within fixed duration
                temp_event[start_idx:start_idx+event_sig.shape[0], :] = event_sig
                event_sig = temp_event
            else:
                # may need to apply some cross fading to prevent discontinuities that can sound as "pop's"
                win = signal.windows.tukey(self.total_frames, 0.005)
                win = np.tile(win.reshape(-1, 1), (1, 32))
                event_sig = event_sig[:self.total_frames] * win

            for ichan in range(self.num_chans):
                conv_sig[:, ichan] += np.convolve(event_sig[:, ichan], rir_sig[:self.audio_FS, ichan], mode='same')
        
            output_filename += f"_{class_id}_{round(coord[1])}_{round(coord[2])}"
        sf.write(os.path.join(dest_dir, f"{output_filename}.wav"), conv_sig, self.audio_FS)


    def spatialize_sinetone_events(self, n_polyphony=1, track_num=0, dest_dir="./output"):
        # generate a copy of the available rirs to be used
        rirs_list, coords_list = copy.deepcopy(self.rirs), copy.deepcopy(self.source_coords)
        # intitalize convolve audio track
        conv_sig = np.zeros((self.total_frames, self.num_chans))
        # Define output filename: <track_num>_<polyphony>_<class_id_1>_<azi_1>_<ele_1>_<class_id_2>_<azi_2>_<ele_2>_... .wav
        output_filename = f"{track_num+1:03d}_polyphony{n_polyphony}"
        class_id = 440
        for _ in range(n_polyphony):
            # get random room impulse reponse
            rir_idx = random.randrange(len(rirs_list))
            rir_sig, coord = rirs_list.pop(rir_idx), coords_list.pop(rir_idx) 
            
            t = np.linspace(0, self.total_duration, int(self.total_frames), endpoint=False)
            freq = random.randint(2666-166, 2666+166) # NOTE: third frequency band
            sinetone = np.sin(2 * np.pi * freq * t)

            event_sig = sinetone.reshape(-1, 1)
            event_sig = np.tile(event_sig, (1, rir_sig.shape[-1]))

            for ichan in range(self.num_chans):
                conv_sig[:, ichan] += np.convolve(event_sig[:, ichan], rir_sig[:self.audio_FS, ichan], mode='same')
        
            output_filename += f"_{class_id}_{round(coord[1])}_{round(coord[2])}"
        sf.write(os.path.join(dest_dir, f"{output_filename}.wav"), conv_sig, self.audio_FS)


if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser(description="Spatial audio synthesizer with YAML config file.")
    parser.add_argument("-c", "--config", required=True, help="Path to the YAML config file")

    # parse command-line arguments
    args = parser.parse_args()

    # load parameters from the specified YAML file
    with open(args.config, "r") as f:
        params = yaml.safe_load(f)

    # initialize destination directory to be same name as the YAML config file
    dest_dir = os.path.splitext(os.path.basename(args.config))[0]
    dest_dir = f"./{dest_dir}"  # assuming the output directory is in the current directory

    # initialze simulation parameters
    rirs, source_coords = get_audio_spatial_data(params["room"])
    audio_tracks_paths = get_audio_tracks()
    n_tracks = params["n_tracks"]
    total_duration = params["total_duration"]
    polyphony = params["polyphony"]
    os.makedirs(dest_dir, exist_ok=True)
    AudioSynth = AudioSynthesizer(rirs, source_coords, audio_tracks_paths, total_duration)

    print("Synthesizing spatial audio...")
    for track_num in tqdm(range(n_tracks)):
        AudioSynth.spatialize_audio_events(polyphony, track_num, dest_dir)
