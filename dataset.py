import numpy as np
import os
import random
import torch
import torchaudio

import torch.nn.functional as F

MAX_WAV_VALUE = 32768.0  # dor added


class BinauralConditionalDataset(torch.utils.data.Dataset):
    def __init__(self, audio_files, len_dataset):
        super().__init__()
        self.audio_files = audio_files
        self.len_dataset = len_dataset

        self.load_from_wavfiles()

    def load_from_wavfiles(self):
        audio_files = self.audio_files

        if len(audio_files) > 0.5 * self.len_dataset:
            if os.path.exists(
                    "n_train.npy") and \
                    os.path.exists("y_train.npy"):


                y_train = np.load("y_train.npy")
                n_train = np.load("n_train.npy")

            self.y = y_train
            self.n = n_train
        else:
            if os.path.exists(
                    "n_valid.npy") and \
                    os.path.exists("y_valid.npy"):

                bsm_with_array_rot_valid_data = np.load("y_valid.npy")
                bsm_with_head_rot_valid_data = np.load("n_valid.npy")

            self.y = bsm_with_array_rot_valid_data
            self.n = bsm_with_head_rot_valid_data



    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, index):
        filename = self.audio_files[index]

        audio = self.y[index]
        audio = audio / MAX_WAV_VALUE
        y = torch.FloatTensor(audio)
        # y = y.T  # dor


        audio = self.n[index]
        audio = audio / MAX_WAV_VALUE
        n = torch.FloatTensor(audio)
        # n = n.T  # dor

        return (y, n, filename)



def from_path(audio_files, len_dataset, params, shuffle=True):

    dataset = BinauralConditionalDataset(audio_files, len_dataset)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=params.batch_size,
        shuffle=shuffle,
        # num_workers=os.cpu_count(),
        num_workers=1,
        sampler=None,
        pin_memory=True,
        drop_last=True)
