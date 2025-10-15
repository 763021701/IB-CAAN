import os
import random
import numpy as np

import torch
from torch import Tensor
from torch.utils.data import Dataset

import librosa

from audiomentations import AddBackgroundNoise, ApplyImpulseResponse
from RawBoost import ISD_additive_noise, LnL_convolutive_noise, SSI_additive_noise,normWav


SPEAKER_STR = [f"LA_00{i:02d}" for i in range(79, 99)]
SPEAKER_2_INT = {s: i for i, s in enumerate(SPEAKER_STR)}
# ATTACK_STR = [f"A0{i}" for i in range(1, 7)] + ['-']
# ATTACK_2_INT = {s: i for i, s in enumerate(ATTACK_STR)}

# or by vocoder
ATTACK_STR = ["WaveNet", "WORLD", "WavConcat", "SpecfiltOLA"]
ATTACK_2_INT = {
    "A01": 0,
    "A02": 1,
    "A03": 1,
    "A04": 2,
    "A05": 1,
    "A06": 3,
    "-": 4
}

def genSpoof_list(dir_meta, is_train=False, is_eval=False):
    d_meta = {}
    d_speaker = {}
    d_attack = {}
    file_list=[]
    attack_label = []  # for WeightedRandomSampler
    with open(dir_meta, 'r') as f:
         l_meta = f.readlines()

    if (is_train):
        for line in l_meta:
             speaker,key,_,attack,label = line.strip().split()
             file_list.append(key)
             d_meta[key] = 1 if label == 'bonafide' else 0
             assert speaker in SPEAKER_2_INT, f"'{speaker}' should in SPEAKER_2_INT"
             d_speaker[key] = SPEAKER_2_INT[speaker]
             assert attack in ATTACK_2_INT, f"'{attack}' should in ATTACK_2_INT"
             d_attack[key] = ATTACK_2_INT[attack]
             attack_label.append(ATTACK_2_INT[attack])
        return d_meta,d_speaker,d_attack,file_list,attack_label

    elif(is_eval):
        for line in l_meta:
            key= line.strip()
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
             speaker,key,_,attack,label = line.strip().split()
             file_list.append(key)
             d_meta[key] = 1 if label == 'bonafide' else 0
             # dev set speaker is not in SPEAKER_2_INT
             assert speaker not in SPEAKER_2_INT, f"'{speaker}' should not in SPEAKER_2_INT"
             d_speaker[key] = 0 # TODO.
             #assert attack not in ATTACK_2_INT, f"'{attack}' should not in ATTACK_2_INT"
             d_attack[key] = 0 # TODO.
        return d_meta,d_speaker,d_attack,file_list


def find_wav_files(rir_dir):
    rir_files = []
    for root, dirs, files in os.walk(rir_dir):
        for f in files:
            if f.endswith('.wav'):
                rir_files.append(os.path.join(root, f))
    return rir_files


class Dataset_ASVspoof2019_train(torch.utils.data.Dataset):
    def __init__(self, args, list_IDs, labels, base_dir,
                 speakers=None, attacks=None, sr=16000):
        self.aug_list = args["aug_list"]
        self.musan_dir = args["musan_dir"]
        self.rir_dir = args["rir_dir"]
        self.rawboost_args = args["rawboost"]

        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.speakers = speakers
        self.attacks = attacks
        self.sr = sr

        self.noise_augment = AddBackgroundNoise(
            sounds_path=self.musan_dir,
            min_snr_db=0,
            max_snr_db=15,
            p=1.0
        )

        self.rir_augment = ApplyImpulseResponse(
            ir_path=self.rir_dir,
            p=1.0,
            leave_length_unchanged=True
        )

    def pad_or_crop(self, audio):
        max_len = 4 * self.sr
        if len(audio) < max_len:
            x_len = len(audio)
            num_repeats = int(max_len / x_len) + 1
            padded_x = np.tile(audio, num_repeats)[:max_len]
            audio = padded_x
        elif len(audio) > max_len:
            start = random.randint(0, len(audio) - max_len)
            audio = audio[start:start+max_len]
        return audio

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, idx):
        utt_id = self.list_IDs[idx]
        target = self.labels[utt_id]
        speaker = self.speakers[utt_id] if self.speakers is not None else 0
        attack = self.attacks[utt_id] if self.attacks is not None else 0
        audio, fs = librosa.load(self.base_dir / (utt_id + '.flac'), sr=self.sr)

        if len(self.rawboost_args["algo_list"]) > 0:
            algo = random.choice(self.rawboost_args["algo_list"])
            audio = process_Rawboost_feature(
                audio, fs, self.rawboost_args, algo)

        # Pad or crop to fixed length
        x_pad = self.pad_or_crop(audio)
        # to torch tensor
        x_inp = torch.tensor(x_pad, dtype=torch.float32)
        return x_inp, target, speaker, attack


class Dataset_ASVspoof2019_dev(Dataset_ASVspoof2019_train):
    def __init__(self, args, list_IDs, labels, base_dir,
                 speakers=None, attacks=None, sr=16000):
        super().__init__(args=args,
                         list_IDs=list_IDs,
                         labels=labels,
                         base_dir=base_dir,
                         speakers=speakers,
                         attacks=attacks,
                         sr=sr)

    def __getitem__(self, idx):
        utt_id = self.list_IDs[idx]
        target = self.labels[utt_id]
        audio, fs = librosa.load(self.base_dir / (utt_id + '.flac'), sr=self.sr)

        # Pad or crop to fixed length
        x_pad = self.pad_or_crop(audio)
        # to torch tensor
        x_inp = torch.tensor(x_pad, dtype=torch.float32)
        return x_inp, target, utt_id


class Dataset_ASVspoof2019_train_for_SWA(Dataset_ASVspoof2019_train):
    def __init__(self, args, list_IDs, labels, base_dir,
                 speakers=None, attacks=None, sr=16000):
        super().__init__(args=args,
                         list_IDs=list_IDs,
                         labels=labels,
                         base_dir=base_dir,
                         speakers=speakers,
                         attacks=attacks,
                         sr=sr)

    def __getitem__(self, idx):
        x_inp, target, speaker, attack = super().__getitem__(idx)
        return x_inp, target, speaker, attack


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


class Dataset_ASVspoof2021_eval(Dataset):
    def __init__(self, list_IDs, base_dir):
        '''self.list_IDs	: list of strings (each string: utt key),
           '''
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut=64600 # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        X, fs = librosa.load(self.base_dir / (utt_id + '.flac'), sr=16000)
        X_pad = pad(X,self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, utt_id


class Dataset_in_the_wild_eval(Dataset):
    def __init__(self, list_IDs, base_dir):
        '''self.list_IDs	: list of strings (each string: utt key),
               '''

        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        X, fs = librosa.load(self.base_dir / utt_id, sr=16000)
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, utt_id


#--------------RawBoost data augmentation algorithms---------------------------##

def process_Rawboost_feature(feature, sr,args,algo):
    # Data process by Convolutive noise (1st algo)
    if algo==1:
        feature = LnL_convolutive_noise(feature,args['N_f'],args['nBands'],args['minF'],args['maxF'],args['minBW'],args['maxBW'],
                args['minCoeff'],args['maxCoeff'],args['minG'],args['maxG'],args['minBiasLinNonLin'],args['maxBiasLinNonLin'],sr)

    # Data process by Impulsive noise (2nd algo)
    elif algo==2:
        feature = ISD_additive_noise(feature, args['P'], args['g_sd'])

    # Data process by coloured additive noise (3rd algo)
    elif algo==3:
        feature = SSI_additive_noise(feature,args['SNRmin'],args['SNRmax'],args['nBands'],args['minF'],args['maxF'],args['minBW'],
                args['maxBW'],args['minCoeff'],args['maxCoeff'],args['minG'],args['maxG'],sr)

    # Data process by all 3 algo. together in series (1+2+3)
    elif algo==4:
        feature = LnL_convolutive_noise(feature,args['N_f'],args['nBands'],args['minF'],args['maxF'],args['minBW'],args['maxBW'],
                args['minCoeff'],args['maxCoeff'],args['minG'],args['maxG'],args['minBiasLinNonLin'],args['maxBiasLinNonLin'],sr)
        feature = ISD_additive_noise(feature, args['P'], args['g_sd'])
        feature = SSI_additive_noise(feature,args['SNRmin'],args['SNRmax'],args['nBands'],args['minF'],
                args['maxF'],args['minBW'],args['maxBW'],args['minCoeff'],args['maxCoeff'],args['minG'],args['maxG'],sr)

    # Data process by 1st two algo. together in series (1+2)
    elif algo==5:
        feature = LnL_convolutive_noise(feature,args['N_f'],args['nBands'],args['minF'],args['maxF'],args['minBW'],args['maxBW'],
                args['minCoeff'],args['maxCoeff'],args['minG'],args['maxG'],args['minBiasLinNonLin'],args['maxBiasLinNonLin'],sr)
        feature = ISD_additive_noise(feature, args['P'], args['g_sd'])


    # Data process by 1st and 3rd algo. together in series (1+3)
    elif algo==6:
        feature = LnL_convolutive_noise(feature,args['N_f'],args['nBands'],args['minF'],args['maxF'],args['minBW'],args['maxBW'],
                args['minCoeff'],args['maxCoeff'],args['minG'],args['maxG'],args['minBiasLinNonLin'],args['maxBiasLinNonLin'],sr)
        feature = SSI_additive_noise(feature,args['SNRmin'],args['SNRmax'],args['nBands'],args['minF'],args['maxF'],args['minBW'],
                args['maxBW'],args['minCoeff'],args['maxCoeff'],args['minG'],args['maxG'],sr)

    # Data process by 2nd and 3rd algo. together in series (2+3)
    elif algo==7:
        feature = ISD_additive_noise(feature, args['P'], args['g_sd'])
        feature = SSI_additive_noise(feature,args['SNRmin'],args['SNRmax'],args['nBands'],args['minF'],args['maxF'],args['minBW'],
                args['maxBW'],args['minCoeff'],args['maxCoeff'],args['minG'],args['maxG'],sr)

    # Data process by 1st two algo. together in Parallel (1||2)
    elif algo==8:
        feature1 = LnL_convolutive_noise(feature,args['N_f'],args['nBands'],args['minF'],args['maxF'],args['minBW'],args['maxBW'],
                 args['minCoeff'],args['maxCoeff'],args['minG'],args['maxG'],args['minBiasLinNonLin'],args['maxBiasLinNonLin'],sr)
        feature2 = ISD_additive_noise(feature, args['P'], args['g_sd'])

        feature_para = feature1+feature2
        feature = normWav(feature_para,0)  #normalized resultant waveform

    # original data without Rawboost processing
    else:
        feature=feature

    return feature
