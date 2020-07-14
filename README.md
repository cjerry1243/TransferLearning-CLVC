# TransferLearning-CLVC

Imlementation of _Transfer Learning from Monolingual ASR to Transcription-free Cross-lingual Voice Conversion._  
We provide our pretrained monolingual uni-directional acoustic model and speaker encoder for reproduction. According to our paper, the uni-directinal content extractor is not  All the VC data are from Voice Conversion Challenge 2020 (http://www.vc-challenge.org/) and all the generated speech are submitted to the challenge for listening review including intra-lingual and cross-lingual VC tasks.  

For more details, please refer to our paper.


## Requirements
- python 3.6
- pytorch >= 1.1
- librosa
- h5py
- scipy
- apex


## Preprocessing
1. Clone this repository.
2. Download data from VCC 2020. (http://www.vc-challenge.org/)
3. Prepare training data for Waveglow vocoder.
```bash
python prepare_h5.py --mode 0
```
This would generate an h5 file that concatenates all the speech for each speaker.
4. Prepare training data for the conversion model.
```bash
python prepare_h5.py --mode 1
```
This would convert the speech into input features, d-vectors, and mel-spectrograms.

## Training Waveglow vocoder
1. Modify the "h5_melfile" in mel2samp.py to specify the h5 files.
2. Modify the config_24k.json for hyperparameters.
3. Run the training script
```bash
python train.py -c config_24k.json
```

## Training the conversion model
1. Modify common/hparams_spk.py for your desired checkpoint directory and hyperparameters.
Be aware that the input dimension can only be 72 or 514, depending on which feature you want to use.
2. Modify train_ppg2mel_spk.py to specify the h5 feature files, and determine which feature you want to use ("ppg" or "ppg_70") in the "PPG2MEL_Dataset_spk" class.
3. Run the training script
```bash
python train_ppg2mel_spk.py
```
Ideally it takes a few days. We stopped at the 30k to 50kth checkpoint. 



## Testing
1. Prepare the evaluation dataset for every source speaker and put the speaker directories under "vcc2020_evaluation".
2. Run the testing script
```bash
python convert_speech_vcc.py -ch "checkpoint_of_conversion_model" -m "ppg_model_you_used" -o "vcc2020_evaluation/output_directory/"
```
converted wav files are in the output directory in the formate of "target_source_wavname.wav"


## Reference
1. guanlongzhao's fac-via-ppg (https://github.com/guanlongzhao/fac-via-ppg)
2. NVIDIA's Waveglow and Tacotron2 (https://github.com/NVIDIA/waveglow, https://github.com/NVIDIA/tacotron2)
3. pytorch's audio (https://github.com/pytorch/audio)
