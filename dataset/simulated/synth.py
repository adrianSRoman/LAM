import os
import cv2
import csv
import random
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm

import scipy
import scipy.signal as signal

from utils import *
from spatializer import *

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

LEAVE_OUT_CLASSES = ["clapping", "telephone", "laughter", "domesticSounds", "footsteps",
                    "doorCupboard", "music", "musicInstrument", "waterTap", "bell", "knock"]

def get_audio_tracks(dataset_path=DCASE_EVENTS, leave_out_classes=LEAVE_OUT_CLASSES, split="train"):
    tracks_list = []
    for event_class in DCASE_SOUND_EVENT_CLASSES.keys():
        if event_class in leave_out_classes:
            continue
        
        for dirpath, dirnames, tracks in os.walk(os.path.join(dataset_path, event_class, split)):
            # ensure no subdirectories
            if len(dirnames) == 0:
                for track in tracks:
                    tracks_list.append(os.path.join(dirpath, track))        


class AudioSynthesizer:
    def __init__(self, rirs, source_coords, audio_tracks_paths, min_duration, max_duration, total_duration):
        self.rirs = rirs # room_impulse responses
        self.source_coords = source_coords
        self.audio_tracks_paths = audio_tracks_paths
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.total_duration = total_duration
        self.channel_num = self.rirs[0].shape[1] # channel count in array
        self.audio_fps = 10 	# 100ms
        self.audio_FS = 24000	# sampling rate (24kHz)
        self.win_size = 512	# window size for spatial convolutions	
        self.stream_total_frames = int(self.total_duration * self.audio_fps)  # Calculate total frames based on total_duration


    def generate_audio_mix_spatialized(self, mix_name):
        self.generate_track_metadata(mix_name)
        audio_mix = np.zeros((self.channel_num, self.audio_FS*self.total_duration), dtype=np.float64)
        for event_data in self.events_history:
            # Load the video file
            audio_sig, sr = librosa.load(event_data['path'], sr=None, mono=None)
            # Extract the audio
            start_idx = int(self.audio_FS * event_data['start_frame']/self.audio_fps)
            duration_samps = int(self.audio_FS * event_data['duration']/self.audio_fps)
            audio_sig = librosa.resample(audio_sig.mean(axis=0), orig_sr=sr, target_sr=self.audio_FS)
            audio_sig = self.spatialize_audio_event(audio_sig, event_data['rir_id'], duration_samps)
            audio_mix[:, start_idx:start_idx+audio_sig.shape[0]] += audio_sig.T #.mean(axis=0) # TODO: [fix] this may cause a 1 frame delay between audio and video streams
            audio_mix /= audio_mix.max()
            sf.write(f'{mix_name}.wav', audio_mix.T, self.audio_FS)


    def spatialize_audio_event(self, track_name):
        # get random room impulse reponse
        rir_idx = random.randrange(len(self.rirs))
        rir = self.rirs[rir_idx]
        coord = self.source_coords[rir_idx]
        # get random sound event to spatialize
        event_idx = random.randrange(len(self.audio_tracks_paths))
        event = self.audio_tracks_paths[event_idx]
        print(coord, event)
        ## get event up to a given duration
        #eventsig = eventsig[:dur_samps+trim_samps]
        ## perform event convolution with rooom impulse response
        #print(rirs.shape, eventsig.shape)
        #return output_signal


# Example usage:
rirs, source_coords = get_audio_spatial_data(aud_fmt="em32", room="METU")
audio_tracks_paths = get_audio_tracks() 
#[os.path.join(directory_path, filename) for filename in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, filename))]
min_duration = 2  # Minimum duration for overlay videos (in seconds)
max_duration = 3  # Maximum duration for overlay videos (in seconds)
total_duration = 15
track_name = "fold1_room002_mix"  # File to save overlay info
AudioSynth = AudioSynthesizer(rirs, source_coords, audio_tracks_paths, min_duration, max_duration, total_duration)

print("Synthesizing spatial audio")
AudioSynth.spatialize_audio_event(track_name)




