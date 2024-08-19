import os
from torch.utils import data
import h5py


class Dataset(data.Dataset):
    def __init__(self, path_to_datasets, freq_band=None, mode="train", leave_out=[]):
        super(Dataset, self).__init__()
        
        # Accept a list of file paths
        self.file_paths = [os.path.join(path_to_datasets, dataset) \
                            for dataset in os.listdir(path_to_datasets) if dataset.endswith(".hdf") and dataset not in leave_out]
        self.freq_band = freq_band
        
        # Open all HDF5 files at initialization and store references
        self.hdf5_files = [h5py.File(file_path, 'r') for file_path in self.file_paths]

        # Calculate the cumulative length of all datasets without loading data into memory
        self.cum_lengths = [0]
        total_length = 0
        for file in self.hdf5_files:
            total_length += len(file['em32'])  # len(em32_matrix)
            self.cum_lengths.append(total_length)

    def __getitem__(self, index):
        # Determine which HDF5 file the index falls into
        for i, length in enumerate(self.cum_lengths[1:], start=1):
            if index < length:
                # Index falls into the i-th file
                file_index = i - 1
                local_index = index - self.cum_lengths[file_index]
                break
        
        # Fetch the data from the correct file and index
        file = self.hdf5_files[file_index]
        em32 = file['em32'][local_index]
        mic = file['mic'][local_index]
        label = file['apgd'][local_index]
        dur = file['dur'][local_index]
        
        if self.freq_band is not None:
            return em32[0, :, :], mic[0, :, :], label[0], dur[0]
        else:
            return em32, mic, label, dur

    def __len__(self):
        return self.cum_lengths[-1]
    
    def __del__(self):
        # Ensure that HDF5 files are closed when the dataset object is deleted
        for file in self.hdf5_files:
            file.close()

