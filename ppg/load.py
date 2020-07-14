import os
import math
import h5py
import wave
import torch
import numpy as np
import scipy.io.wavfile as wav
from .torchaudio.transforms import MFCC

from tqdm import tqdm
from collections import Counter
from torch.utils.data import Dataset


rate = 16000
n_mfcc = 40
n_fft = 512
win_length = 400
hop_length = 160
trim_mode = 2

leftpad = 7
rightpad = 7

MFCC_Model = MFCC(rate, n_mfcc=n_mfcc, log_mels=True,
                  melkwargs={"n_fft": n_fft, "win_length": win_length, "hop_length": hop_length,
                             "window_fn": torch.hann_window, "symmetry": True, "center": False})


def get_mfcc_features(path, pad_zero=False):
    rate, sig = wav.read(path)
    assert rate == 16000, "Error. Sample rate for input wav must be 16000!"
    if pad_zero:
        sig = np.concatenate((np.zeros([n_fft + hop_length * (leftpad - 3),]), sig, np.zeros([n_fft + hop_length * (rightpad - 3),])))
    mfccs = MFCC_Model(torch.FloatTensor(sig/32768.0).unsqueeze(0)).squeeze(0).transpose(0, 1) # (t, n_mfcc)
    return mfccs, len(sig), rate


# def get_mfcc_stats():
#     mean = torch.load("ppg/mean_mfcc_40.pth")
#     std = torch.load("ppg/std_mfcc_40.pth")
#     return mean, std

# mean, std = get_mfcc_stats()


if __name__ == "__main__":
    pass








