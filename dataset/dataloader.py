from torch.utils import data
import h5py

class Dataset(data.Dataset):
    def __init__(self, dataset, freq_band=None, mode="train"):
        super(Dataset, self).__init__()
        
        self.file_path = dataset
        self.freq_band = freq_band
        self.file = h5py.File(self.file_path, 'r')
        self.audio_data = self.file['labels'][64:64+64*8] # MIC audio data (nframes, nsamps, nch)
        self.label_data = self.file['labels'][64:64+64*8] # visibility graph matrices (nframes, nbands, nch, nch)

    def __getitem__(self, index):

        data =  self.audio_data[index]
        
        if self.freq_band is not None:
            return data[self.freq_band, :, :]
        else:
            return data

    def __len__(self):
        return len(self.audio_data)
