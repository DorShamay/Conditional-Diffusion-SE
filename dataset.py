import numpy as np
import os
import random
import torch
import torchaudio

import torch.nn.functional as F

MAX_WAV_VALUE = 32768.0  # dor added


class BinauralConditionalDataset(torch.utils.data.Dataset):
    def __init__(self, audio_files, len_dataset, predict_mean_condition=False):
        super().__init__()
        self.audio_files = audio_files
        self.len_dataset = len_dataset

        self.predict_mean_condition = predict_mean_condition


        self.load_from_wavfiles()

    def load_from_wavfiles(self):
        audio_files = self.audio_files

        if len(audio_files) > 0.5 * self.len_dataset:
            if os.path.exists(
                    "n_train.npy") and \
                    os.path.exists("y_train.npy"):


                bsm_with_array_rot_train_data = np.load("y_train.npy")
                bsm_with_head_rot_train_data = np.load("n_train.npy")

            self.bsm_with_array_rot_data = bsm_with_array_rot_train_data
            self.bsm_with_head_rot_data = bsm_with_head_rot_train_data
        else:
            if os.path.exists(
                    "n_valid.npy") and \
                    os.path.exists("y_valid.npy"):

                bsm_with_array_rot_valid_data = np.load("y_valid.npy")
                bsm_with_head_rot_valid_data = np.load("n_valid.npy")

            self.bsm_with_array_rot_data = bsm_with_array_rot_valid_data
            self.bsm_with_head_rot_data = bsm_with_head_rot_valid_data



    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, index):
        filename = self.audio_files[index]

        audio = self.bsm_with_array_rot_data[index]
        audio = audio / MAX_WAV_VALUE
        audio_bsm_with_array_rot = torch.FloatTensor(audio)
        # audio_bsm_with_array_rot = audio_bsm_with_array_rot.T  # dor


        audio = self.bsm_with_head_rot_data[index]
        audio = audio / MAX_WAV_VALUE
        audio_bsm_with_head_rot = torch.FloatTensor(audio)
        # audio_bsm_with_head_rot = audio_bsm_with_head_rot.T  # dor

        return (audio_bsm_with_array_rot, audio_bsm_with_head_rot, filename)



def from_path(audio_files, len_dataset, params, is_distributed=False):

    dataset = BinauralConditionalDataset(audio_files, len_dataset,
                                         predict_mean_condition=getattr(params, "predict_mean_condition", False))

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=params.batch_size,
        shuffle=not is_distributed,
        # num_workers=os.cpu_count(),
        num_workers=1,
        sampler=None,
        pin_memory=True,
        drop_last=True)
