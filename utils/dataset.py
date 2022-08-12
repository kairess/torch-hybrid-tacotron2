import os
import torch
import random
import numpy as np
from text import text_to_sequence
from hparams import hparams as hps
from torch.utils.data import Dataset
from utils.audio import load_wav, melspectrogram, spectrogram

random.seed(0)


def files_to_list(fdir):
    f_list = []
    with open(os.path.join(fdir, 'metadata.csv'), encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            wav_path = os.path.join(fdir, 'wavs', '%s.wav' % parts[0])
            f_list.append([wav_path, parts[2]])
    return f_list


def load_metadata(fdir, findex, sindex):
    metadata = []
    with open(os.path.join(fdir, 'metadata.csv'), encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            filename = parts[findex]
            transcript = parts[sindex].lower()
            wav_path = os.path.join(fdir, 'wavs', '%s.wav' % filename)
            metadata.append([wav_path, transcript])
    return metadata


def load_metadata_v2(fdir, mindex, lindex, sindex):
    traindata_dir = os.path.join(fdir, 'train_data_%d' % hps.sample_rate)
    mel_dir = os.path.join(traindata_dir, 'mels')
    linear_dir = os.path.join(traindata_dir, 'linears')

    metadata = []
    with open(os.path.join(traindata_dir, 'train_metadata.csv'), encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            mel_path = os.path.join(mel_dir, parts[mindex])
            linear_path = os.path.join(linear_dir, parts[lindex])
            transcript = parts[sindex].lower()
            metadata.append([mel_path, linear_path, transcript])
    return metadata


class ljdataset(Dataset):
    def __init__(self, fdir):
        self.f_list = files_to_list(fdir)
        random.shuffle(self.f_list)

    def get_mel_text_pair(self, filename_and_text):
        filename, text = filename_and_text[0], filename_and_text[1]
        text = self.get_text(text)
        mel, linear = self.get_mel_linear_pair(filename)
        # print('mel size : ', mel.size()) # mel size :  torch.Size([80, 720])
        # print('linear size : ', linear.size()) # linear size :  torch.Size([1025, 720])
        return (text, mel, linear)

    def get_mel(self, filename):
        wav = load_wav(filename)
        mel = melspectrogram(wav).astype(np.float32)
        return torch.Tensor(mel)

    def get_mel_linear_pair(self, filename):
        wav = load_wav(filename)
        mel = melspectrogram(wav).astype(np.float32)
        linear = spectrogram(wav).astype(np.float32)
        return torch.Tensor(mel), torch.Tensor(linear)

    def get_text(self, text):
        text_norm = torch.IntTensor(text_to_sequence(text, hps.text_cleaners))
        return text_norm

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.f_list[index])

    def __len__(self):
        return len(self.f_list)


class ljcollate():
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        num_linears = batch[0][2].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        linear_padded = torch.FloatTensor(len(batch), num_linears, max_target_len)
        linear_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            linear = batch[ids_sorted_decreasing[i]][2]
            linear_padded[i, :, :linear.size(1)] = linear
            gate_padded[i, mel.size(1) - 1:] = 1
            output_lengths[i] = mel.size(1)

        return text_padded, input_lengths, mel_padded, linear_padded, gate_padded, output_lengths


class KssDataset(Dataset):
    def __init__(self, fdir):
        self.metadata = load_metadata(fdir, 0, 2)
        random.shuffle(self.metadata)

    def get_item(self, filename_and_text):
        filename, text = filename_and_text[0], filename_and_text[1]
        text = self.get_text(text)
        mel, linear = self.get_mel_linear_pair(filename)
        return (text, mel, linear)

    def get_mel_linear_pair(self, filename):
        wav = load_wav(filename)
        mel = melspectrogram(wav).astype(np.float32)
        linear = spectrogram(wav).astype(np.float32)
        return torch.Tensor(mel), torch.Tensor(linear)

    def get_text(self, text):
        text_norm = torch.IntTensor(text_to_sequence(text, hps.text_cleaners))
        return text_norm

    def __getitem__(self, index):
        return self.get_item(self.metadata[index])

    def __len__(self):
        return len(self.metadata)


class KssDatasetV2(Dataset):
    def __init__(self, fdir):
        self.metadata = load_metadata_v2(fdir, 0, 1, 2)
        random.shuffle(self.metadata)

    def get_item(self, items):
        mel_path, linear_path, transcript = items[0], items[1], items[2]
        text = self.get_text(transcript)
        mel, linear = self.get_mel_linear_pair(mel_path, linear_path)
        return (text, mel, linear)

    def get_mel_linear_pair(self, mel_path, linear_path):
        mel = torch.load(mel_path)
        linear = torch.load(linear_path)
        return mel, linear

    def get_text(self, text):
        text_norm = torch.IntTensor(text_to_sequence(text, hps.text_cleaners))
        return text_norm

    def __getitem__(self, index):
        return self.get_item(self.metadata[index])

    def __len__(self):
        return len(self.metadata)


class KssCollate():
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        num_linears = batch[0][2].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        linear_padded = torch.FloatTensor(len(batch), num_linears, max_target_len)
        linear_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            linear = batch[ids_sorted_decreasing[i]][2]
            linear_padded[i, :, :linear.size(1)] = linear
            gate_padded[i, mel.size(1) - 1:] = 1
            output_lengths[i] = mel.size(1)

        return text_padded, input_lengths, mel_padded, linear_padded, gate_padded, output_lengths
