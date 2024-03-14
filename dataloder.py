import torch.utils.data as data
import h5py

class DatasetLoader(data.Dataset):
    def __init__(self, file_path, upscale_factor=8):
        super(DatasetLoader, self).__init__()
        self.upscale_factor = upscale_factor

        self.file_path = file_path
        self.file = h5py.File(self.file_path, 'r')
        self.audio_data = self.file['labels'][[30, 65, 89, 249, 299]] # MIC audio data (nframes, nsamps, nch)
        self.label_data = self.file['labels'][[30, 65, 89, 249, 299]] # visibility graph matrices (nframes, nbands, nch, nch)

    def __getitem__(self, index):

        data =  self.audio_data[index]
        label = self.label_data[index]
        return data, label

    def __len__(self):
        return len(self.audio_data)
