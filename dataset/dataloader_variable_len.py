from torch.utils import data
import h5py

class Dataset(data.Dataset):
    def __init__(self, dataset, freq_band=None, mode="train"):
        super(Dataset, self).__init__()
        
        self.file_path = dataset
        print("dataset filepath", dataset)
        self.freq_band = freq_band
        self.file = h5py.File(self.file_path, 'r')
        self.audio_data = self.file['em32'] # MIC audio data (nframes, nsamps, nch)
        self.label_data = self.file['apgd'] # visibility graph matrices (nframes, nbands, nch, nch)
        self.dur_data = self.file['dur'] # audio frame duration

    def __getitem__(self, index):

        data = self.audio_data[index]
        label = self.label_data[index]
        dur = self.dur_data[index]
        if self.freq_band is not None:
            return data[0, :, :], label[0], dur[0]
        else:
            return data, label, dur

    def __len__(self):
        return len(self.audio_data)