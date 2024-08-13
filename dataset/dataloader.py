from torch.utils import data
import h5py

class Dataset(data.Dataset):
    def __init__(self, dataset, freq_band=None, mode="train"):
        super(Dataset, self).__init__()
        
        self.file_path = dataset
        self.freq_band = freq_band
        self.file = h5py.File(self.file_path, 'r')
        self.em32_matrix = self.file['em32']    # eigenmike32 visibility matrix (nframes, nbands, nch nch)
        self.mic_matrix = self.file['mic']      # tetra mic visibility matrix (nframs, nbands, nch nch)
        self.label_data = self.file['apgd']     # visibility graph matrices (nframes, nbands, Npx)
        self.dur_data = self.file['dur']        # audio frame duration

    def __getitem__(self, index):

        em32 = self.em32_matrix[index]
        mic = self.mic_matrix[index]
        label = self.label_data[index]
        dur = self.dur_data[index]
        if self.freq_band is not None:
            return em32[0,:,:], mic[0,:,:], label[0], dur[0]
        else:
            return em32, mic, label, dur

    def __len__(self):
        return len(self.em32_matrix)
