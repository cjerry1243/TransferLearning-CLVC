import os
import sys
import h5py
import torch
import argparse
import librosa
import numpy as np
from tqdm import tqdm
from scipy.io.wavfile import read

from ppg.load import *
from mel2samp import *


def frame_inference(wavpath, model, use_cuda=True, pad_zero=False, sig=None):
    if sig is None:
        mfccs, _, _ = get_mfcc_features(wavpath, pad_zero) # (t, n_mfcc)
    else:
        mfccs = MFCC_Model(torch.FloatTensor(sig/32768.0).unsqueeze(0)).squeeze(0).transpose(0, 1) # (t, n_mfcc)
    mfccs = (mfccs-mean)/std
    input_x = mfccs.as_strided(size=[mfccs.shape[0]-leftpad-rightpad, n_mfcc*(leftpad+rightpad+1)], stride=[n_mfcc, 1])
    if use_cuda:
        model = model.cuda()
        input_x = input_x.cuda()
    with torch.no_grad():
        ppg = model(input_x.unsqueeze(0))[0].squeeze(0) # try not to add softmax
    return ppg
    # sequence = torch.argmax(out, dim=-1).detach().cpu().tolist()
    # return sequence


def creat_vcc_audio_h5():
    wav_root_path = "vcc2020_training"
    spks = ['SEF1', 'SEF2', 'SEM1', 'SEM2', 'TEF1', 'TEF2', 'TEM1', 'TEM2', 'TFF1', 'TFM1', 'TGF1', 'TGM1', 'TMF1', 'TMM1']
    with h5py.File("VCC_audio_24k_16k.h5", "w") as h5:
        for i, spk in tqdm(enumerate(spks)):
            h5.create_group(str(i))
            wavfiles = os.listdir(os.path.join(wav_root_path, spk))
            audio = np.zeros([0,], dtype=np.float32)
            for wn in wavfiles:
                sig, _ = librosa.load(os.path.join(wav_root_path, spk, wn), 24000)
                audio = np.concatenate((audio, sig))
            audio_16k = np.clip(librosa.resample(audio, 24000, 16000), -1, 1)
            h5[str(i)].create_dataset("24k", data=(audio*32768).astype(np.int16), dtype=np.int16)
            h5[str(i)].create_dataset("16k", data=(audio_16k*32768).astype(np.int16), dtype=np.int16)



def create_vcc_spk_h5():
    use_cuda = True

    with open("config_24k.json") as f:
        data = f.read()
    data_config = json.loads(data)["data_config"]
    mel2samp_24k = Mel2Samp(**data_config)

    import math
    import librosa
    from tqdm import tqdm
    from spk_embedder.embedder import SpeechEmbedder
    embedder = SpeechEmbedder()
    embedder.load_state_dict(torch.load("spk_embedder/embedder.pt"))
    embedder = embedder.cuda().eval() if use_cuda else embedder.cpu().eval()


    mel_basis = librosa.filters.mel(sr=16000, n_fft=512, n_mels=40)
    def get_mel(y):
        y = librosa.core.stft(y, n_fft=512,
                              hop_length=160,
                              win_length=400,
                              window='hann')
        mel = np.log10(np.dot(mel_basis, np.abs(y) ** 2) + 1e-6)
        return mel

    # model = torch.jit.load(os.path.join("ppg", "trace512_77_correct1_epoch-352_feature.pth"))
    model = torch.jit.load(os.path.join("ppg", "trace512xbi_77_correct1_epoch-281_feature.pth"))
    model = model.cuda().eval() if use_cuda else model.cpu().eval()

    # model_ppg = torch.jit.load(os.path.join("ppg", "trace512_77_correct1_epoch-352.pth"))
    model_ppg = torch.jit.load(os.path.join("ppg", "trace512xbi_77_correct1_epoch-281.pth"))
    model_ppg = model_ppg.cuda().eval() if use_cuda else model_ppg.cpu().eval()

    min_audio_seconds = 10
    num_spk_dvecs = 30
    h5_audio = h5py.File("VCC_audio_24k_16k.h5", "r")
    # with h5py.File("VCC_ppg512_16kmel_24kmel_spk.h5", "w") as h5:
    with h5py.File("VCC_ppgbi1024_16kmel_24kmel_spk.h5", "w") as h5:
        for i in tqdm(range(14)):
            audio_24k = h5_audio[str(i)]["24k"][:]/MAX_WAV_VALUE
            audio = h5_audio[str(i)]["16k"][:]/MAX_WAV_VALUE

            dvec = np.zeros([num_spk_dvecs, 256], dtype=np.float32)
            for j in range(num_spk_dvecs):
                st = np.random.randint(0, len(audio)-16000*min_audio_seconds)
                dvec[j] = embedder(torch.FloatTensor(get_mel(audio[st:st+16000*min_audio_seconds])).cuda()).cpu().detach().numpy()

            log_energy = np.zeros([math.ceil(len(audio) / 160), ], dtype=np.float32)
            zcr = np.zeros_like(log_energy, dtype=np.float32)

            for frame_id, seg_start in enumerate(range(0, len(audio), 160)):
                if seg_start + 512 > len(audio):
                    break
                seg = audio[seg_start: seg_start + 512]
                log_energy[frame_id] = np.log(np.sum(seg ** 2) + 1e-8)
                sign_seg = np.sign(seg)
                zcr[frame_id] = np.mean(sign_seg[:-1] != sign_seg[1:])

            log_energy = log_energy[7:frame_id - 7]
            zcr = zcr[7:frame_id - 7]
            ppg = frame_inference("", model, use_cuda, sig=audio * MAX_WAV_VALUE).cpu().detach().numpy()
            ppg_70 = frame_inference("", model_ppg, use_cuda, sig=audio * MAX_WAV_VALUE).cpu().detach().numpy() # (T1, 70)

            if log_energy.shape[0] != ppg.shape[0] or log_energy.shape[0] != ppg_70.shape[0]:
                print("Error calculating intonation features. Found number of frames mismatched")
                exit()

            mel_24k = mel2samp_24k.get_mel(torch.FloatTensor(audio_24k) * MAX_WAV_VALUE).t()
            mel_24k = mel_24k[10:10+ppg.shape[0]].numpy()

            if mel_24k.shape[0] != ppg.shape[0]:
                print("Error. Found number of frames mismatched in mel and ppg")
                exit()

            utterance_gp = h5.create_group(str(i))
            utterance_gp.create_dataset("zcr", data=zcr, dtype=np.float32)
            utterance_gp.create_dataset("log_energy", data=log_energy, dtype=np.float32)
            utterance_gp.create_dataset("ppg", data=ppg, dtype=np.float32)
            utterance_gp.create_dataset("mel_24k", data=mel_24k, dtype=np.float32)
            utterance_gp.create_dataset("dvec", data=dvec, dtype=np.float32)
            utterance_gp.create_dataset("ppg_70", data=ppg_70, dtype=np.float32)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=int, required=True)
    args = parser.parse_args()
    if args.mode == 0:
    	creat_vcc_audio_h5()
    elif args.mode == 1:
    	create_vcc_spk_h5()
    else:
    	print("Error. Invalid mode.")

    pass

