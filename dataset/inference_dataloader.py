import os
from torch.utils.data import Dataset
import librosa


class InferenceDataset(Dataset):
    def __init__(self, dataset, limit=None, offset=0, sample_length=16384):
        """Construct dataset for inference.
        Args:
            dataset (str): The path of the dataset wavefiles.

        Return:
            (audio_signal, filename)
        """
        super(InferenceDataset, self).__init__()
        dataset_list = [os.path.join(dataset, wavfile) for wavfile in os.path.listdir(dataset) if wavfile.endswith(".wav")]

        self.length = len(dataset_list)
        self.dataset_list = dataset_list

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        clip_path = self.dataset_list[item]
        name = os.path.splitext(os.path.basename(clip_path))[0] # get filename

        mixture, _ = librosa.load(os.path.abspath(os.path.expanduser(mixture_path)), sr=None)

        return mixture.reshape(1, -1), name
