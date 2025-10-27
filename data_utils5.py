import numpy as np
import soundfile as sf
import torch
from torch import Tensor
from torch.utils.data import Dataset
from audiomentations import Compose, AddBackgroundNoise, ApplyImpulseResponse
import random


ATTACK_STR = [f"A0{i}" for i in range(1, 9)] + ['bonafide']
ATTACK_2_INT = {s: i for i, s in enumerate(ATTACK_STR)}

def genSpoof_list(dir_meta, is_train=False, is_eval=False):
    d_meta = {}
    d_attack = {}
    file_list = []
    with open(dir_meta, "r") as f:
        l_meta = f.readlines()

    if is_train:
        for line in l_meta:
            _, key, _, _, _, _, _, attack, label, _ = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
            assert attack in ATTACK_2_INT, f"'{attack}' should in ATTACK_2_INT"
            d_attack[key] = ATTACK_2_INT[attack]
        return d_meta, d_attack, file_list

    elif is_eval:
        for line in l_meta:
            _, key, _, _, _, _, _, _, label, _ = line.strip().split(" ")
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
            _, key, _, _, _, _, _, attack, label, _ = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
            #assert attack not in ATTACK_2_INT, f"'{attack}' should not in ATTACK_2_INT"
            d_attack[key] = 0 # TODO.
        return d_meta, d_attack, file_list


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def pad_random(x: np.ndarray, max_len: int = 64600):
    x_len = x.shape[0]
    # if duration is already long enough
    if x_len >= max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]

    # if too short
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x


class Dataset_ASVspoof5_train(Dataset):
    def __init__(self, args, list_IDs, labels, base_dir, attacks=None):
        """self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)"""
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.attacks = attacks
        self.cut = 64600  # take ~4 sec audio (64600 samples)
        self.sr = 16000
        self.aug_list = args["aug_list"]
        print("Data augmentation:", self.aug_list)
        self.musan_dir = args["musan_dir"]
        self.rir_dir = args["rir_dir"]

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

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        y = self.labels[key]
        attack = self.attacks[key] if self.attacks is not None else 0
        audio, _ = sf.read(str(self.base_dir / f"{key}.flac"), dtype='float32')

        if len(self.aug_list) > 0:
            aug = random.choice(self.aug_list)
            if aug == "musan":
                audio = self.noise_augment(samples=audio, sample_rate=self.sr)
            if aug == "rir":
                audio = self.rir_augment(samples=audio, sample_rate=self.sr)

        # Pad or crop to fixed length
        x_pad = pad_random(audio)
        # to torch tensor
        x_inp = Tensor(x_pad)

        return x_inp, y, 0, attack


class Dataset_ASVspoof5_train_for_SWA(Dataset_ASVspoof5_train):
    def __init__(self, args, list_IDs, labels, base_dir, attacks=None):
        super().__init__(
            args=args,
            list_IDs=list_IDs,
            labels=labels,
            base_dir=base_dir,
            attacks=attacks
        )

    def __getitem__(self, idx):
        x_inp, target, speaker, attack = super().__getitem__(idx)
        return x_inp, target, speaker, attack


class Dataset_ASVspoof5_dev(Dataset_ASVspoof5_train):
    def __init__(self, args, list_IDs, labels, base_dir, attacks=None):
        super().__init__(
            args=args,
            list_IDs=list_IDs,
            labels=labels,
            base_dir=base_dir,
            attacks=attacks
        )

    def __getitem__(self, index):
        key = self.list_IDs[index]
        y = self.labels[key]
        attack = self.attacks[key] if self.attacks is not None else 0
        audio, _ = sf.read(str(self.base_dir / f"{key}.flac"), dtype='float32')

        # Pad or crop to fixed length
        x_pad = pad(audio)
        # to torch tensor
        x_inp = Tensor(x_pad)

        return x_inp, y, key



class Dataset_ASVspoof5_eval(Dataset):
    def __init__(self, list_IDs, base_dir):
        """self.list_IDs	: list of strings (each string: utt key),
        """
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = sf.read(str(self.base_dir / f"{key}.flac"), dtype="float32")
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, key
