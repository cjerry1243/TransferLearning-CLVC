
import os
import h5py
import math
import random
import argparse
import json
import torch
import torch.utils.data
import sys
import numpy as np
from scipy.io.wavfile import read

from common.layers import TacotronSTFT

MAX_WAV_VALUE = 32768.0

def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding='utf-8') as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    return files

def load_wav_to_torch(full_path):
    """
    Loads wavdata into torch array
    """
    sampling_rate, data = read(full_path)
    return torch.from_numpy(data).float(), sampling_rate


class Mel2Samp(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """
    def __init__(self, training_files, segment_length, filter_length,
                 hop_length, win_length, sampling_rate, mel_fmin, mel_fmax,
                 h5_melfile=""):
        self.audio_files = files_to_list(training_files)
        self.stft = TacotronSTFT(filter_length=filter_length,
                                 hop_length=hop_length,
                                 win_length=win_length,
                                 sampling_rate=sampling_rate,
                                 mel_fmin=mel_fmin, mel_fmax=mel_fmax)
        self.segment_framelength = math.ceil(segment_length / hop_length)
        self.segment_length = self.segment_framelength * hop_length
        self.hop_length = hop_length
        self.sampling_rate = sampling_rate
        self.h5_melfile = h5_melfile
        self.h5_mel = None

    def get_mel(self, audio):
        audio_norm = audio / MAX_WAV_VALUE
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)
        return melspec


    def __getitem__(self, index):
        if self.h5_mel is None:
            self.h5_mel = h5py.File(self.h5_melfile, "r")

        audio_gp = self.h5_mel[str(index)]["24k"]
        
        audio_start = random.randint(0, audio_gp.shape[0] - self.segment_length)
        audio = torch.FloatTensor(audio_gp[audio_start : audio_start + self.segment_length])

        mel = self.get_mel(audio)

        audio = audio / MAX_WAV_VALUE
        return (mel, audio)

    def __len__(self):
        return len(self.audio_files)

