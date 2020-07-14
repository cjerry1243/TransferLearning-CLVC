import os
import math
import h5py
import torch
import numpy as np
import argparse
import librosa
import time
from mel2samp import files_to_list, MAX_WAV_VALUE
from denoiser import Denoiser
from scipy.io.wavfile import read, write
from prepare_h5 import frame_inference
from common.hparams_spk import create_hparams
from common.model import Tacotron2_multispeaker
from train_ppg2mel_spk import *
from spk_embedder.embedder import SpeechEmbedder


def get_mel(y):
	y = librosa.core.stft(y, n_fft=512, hop_length=160, win_length=400, window='hann')
	mel = np.log10(np.dot(mel_basis, np.abs(y) ** 2) + 1e-6)
	return mel


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
# parser.add_argument('-w', "--wavpath", type=str, required=True)
# parser.add_argument('-r', "--reference", type=str, required=True)
parser.add_argument('-ch', "--checkpoint_path", type=str, required=True)
parser.add_argument('-wg', "--waveglow", type=str, default="checkpoints_24k_VCC2020/waveglow_it248000.pt")
parser.add_argument('-m', "--model", type=str, default=os.path.join("ppg", "trace512xbi_77_correct1_epoch-281_feature.pth"))
parser.add_argument('-s', "--sigma", type=float, default=0.8)
parser.add_argument('-o', "--outputs", type=str, required=True)
parser.add_argument("--sampling_rate", default=24000, type=int)
parser.add_argument("--cuda", default=True, type=bool)
parser.add_argument("--is_fp16", default=True, type=float)
parser.add_argument("-d", "--denoiser_strength", default=0.08, type=float, help='Removes model bias. Start with 0.1 and adjust')
parser.add_argument("-dmd", "--dvec_mode", default=1, type=int, help='dvec mode, '
                                                                     '0: from speaker embedder, '
                                                                     '1: from dictionary, '
                                                                     '2: one-hot, '
                                                                     '3: zeros, single speaker')
args = parser.parse_args()
os.makedirs(args.outputs, exist_ok=True)

model = torch.jit.load(args.model).eval()

### load ppg2mel model
hparams = create_hparams()
torch.manual_seed(hparams.seed)
torch.cuda.manual_seed(hparams.seed)
ppg2mel_model = Tacotron2_multispeaker(hparams)
if args.checkpoint_path is not None:
	ppg2mel_model.load_state_dict(torch.load(args.checkpoint_path)['state_dict'])
ppg2mel_model.cuda().eval()

### load waveglow model
waveglow = torch.load(args.waveglow)['model']
waveglow = waveglow.remove_weightnorm(waveglow)
waveglow = waveglow.cuda().eval()

if args.is_fp16:
	from apex import amp
	waveglow, _ = amp.initialize(waveglow, [], opt_level="O3")

if args.denoiser_strength > 0:
	denoiser = Denoiser(waveglow).cuda()



if args.dvec_mode != 1:
	print("Invalid dvec mode for VCC2020, only 1 is supported.")
h5 = h5py.File("VCC_ppgbi1024_16kmel_24kmel_spk.h5", "r")
wav_root_path = "vcc2020_evaluation"
wav_root_path = "vcc2020_dev"
spks = ['SEF1', 'SEF2', 'SEM1', 'SEM2', 'TEF1', 'TEF2', 'TEM1', 'TEM2', 'TFF1', 'TFM1', 'TGF1', 'TGM1', 'TMF1', 'TMM1']
for sid in range(0, 4):
	source_name = spks[sid]
	source_folder = os.path.join(wav_root_path, source_name)
	wavnames = os.listdir(source_folder)
	for wi, wavname in enumerate(wavnames):
		wavpath = os.path.join(source_folder, wavname)
		source, _ = librosa.load(wavpath, 16000)
		source = source * 32768
		source = np.concatenate((np.zeros([512 + 160 * (7 - 3), ]),
		                         source,
		                         np.zeros([512 + 160 * (7 - 3), ])))
		amp = 32768 * 0.8 / np.max(np.abs(source))
		# interval = librosa.effects.split(source / 32768, 20)
		# for ii in interval:
		# 	seg_amp = 32768 * 0.8 / np.max(np.abs(source[ii[0]: ii[1]]))
		# 	source[ii[0]: ii[1]] = source[ii[0]: ii[1]] * seg_amp ** 0.5
		ppg = frame_inference(wavpath, model, use_cuda=args.cuda, sig=source).cuda().detach()  # (T1, D)
		zcr = np.zeros([math.ceil(len(source) / 160), ], dtype=np.float32)
		log_energy = np.zeros([math.ceil(len(source) / 160), ], dtype=np.float32)
		for frame_id, seg_start in enumerate(range(0, len(source), 160)):
			if seg_start + 512 > len(source):
				break
			seg = source[seg_start: seg_start + 512] / 32768.0
			log_energy[frame_id] = np.log(amp ** 0.5 * np.sum(seg ** 2) + 1e-8)
			# log_energy[frame_id] = np.log(np.sum(seg ** 2) + 1e-8)
			sign_seg = np.sign(seg)
			zcr[frame_id] = np.mean(sign_seg[:-1] != sign_seg[1:])
		zcr = zcr[7:frame_id - 7]
		log_energy = log_energy[7:frame_id - 7]
		if log_energy.shape[0] != ppg.shape[0]:
			print("Error. Frame length mismatched. {} != {}".format(ppg.shape[0], log_energy.shape[0]))
			exit()
		ppg = torch.cat((ppg,
		                 torch.log(torch.FloatTensor(zcr).cuda() + 1e-8).unsqueeze(1),
		                 torch.FloatTensor(log_energy).cuda().unsqueeze(1)), dim=-1)
		ppg = ppg.unsqueeze(0).transpose(1, 2)

		for tid in range(4, 14):
			target_name = spks[tid]
			# dvec = torch.FloatTensor(h5[str(tid)]["dvec"][np.random.randint(0, h5[str(tid)]["dvec"].shape[0]), :]).cuda()
			dvec = torch.mean(torch.FloatTensor(h5[str(tid)]["dvec"][:]), dim=0).cuda()

			print("Source: {}, {}/{}, Wav: {}, Target: {}     ".format(source_name, wi+1, len(wavnames), wavname, target_name), end="\r")
			with torch.no_grad():
				y_pred = ppg2mel_model.inference(ppg, dvec.reshape(1, -1, 1).repeat(1, 1, ppg.shape[2]))
				mel_outputs_before_postnet, mel, gate_outputs, alignments = y_pred
				mel = mel.half() if args.is_fp16 else mel
				audio = waveglow.infer(mel, sigma=args.sigma)
				if args.denoiser_strength > 0:
					audio = denoiser(audio.cuda(), args.denoiser_strength)
				audio = audio * 32768

			audio = audio.squeeze().cpu().numpy().astype('int16')
			audio_path = os.path.join(args.outputs, "{}_{}_{}".format(target_name, source_name, wavname))
			write(audio_path, args.sampling_rate, audio)
		# exit()

print()



if __name__ == "__main__":
	pass