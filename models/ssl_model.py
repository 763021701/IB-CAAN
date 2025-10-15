import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import fairseq

_pretrained_model_path = '/home/ubuntu/workspace/project/IB-CAAN/pretrained_model'

class Wav2vec2_XLSR_300m(nn.Module):
    def __init__(self, device, cp_path = os.path.join(_pretrained_model_path, 'xlsr2_300m.pt')):
        super(Wav2vec2_XLSR_300m, self).__init__()

        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0]
        self.device = device
        self.out_dim = 1024 # for xlsr2_300m
        self.num_layers = 24
        return

    def extract_feat(self, input_data, layer = 23):

        # put the model to GPU if it not there
        #if next(self.model.parameters()).device != input_data.device \
        #   or next(self.model.parameters()).dtype != input_data.dtype:
        #    self.model.to(input_data.device, dtype=input_data.dtype)
        #    # self.model.train()


        if True:
            # input should be in shape (batch, length)
            if input_data.ndim == 3:
                input_tmp = input_data[:, :, 0]
            else:
                input_tmp = input_data

            # Extract features from each layer
            emb = self.model.extract_features(input_tmp, padding_mask = None, layer = layer)
            output = emb['x'] # [batch, length, dim], the last layer of transformer encoder

            emb = [l[0].transpose(0, 1) for l in emb['layer_results']] # emb[i].shape as same as output.shape
            emb = torch.cat(emb, dim=1).reshape(input_data.shape[0], layer+1, -1, self.out_dim) # list of (bs, frame_num, feat) -> (bs, layer_num, frame_num, feat)

        return output, emb


class Wav2vec2_XLSR_1b(nn.Module):
    def __init__(self, device, cp_path = os.path.join(_pretrained_model_path, 'xlsr2_960m_1000k.pt')):
        super(Wav2vec2_XLSR_1b, self).__init__()

        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0]
        self.device = device
        self.out_dim = 1280
        self.num_layers = 48
        return

    def extract_feat(self, input_data, layer = 47):

        # put the model to GPU if it not there
        if next(self.model.parameters()).device != input_data.device \
           or next(self.model.parameters()).dtype != input_data.dtype:
            self.model.to(input_data.device, dtype=input_data.dtype)
            # self.model.train()


        if True:
            # input should be in shape (batch, length)
            if input_data.ndim == 3:
                input_tmp = input_data[:, :, 0]
            else:
                input_tmp = input_data

            # Extract features from each layer
            emb = self.model.extract_features(input_tmp, padding_mask = None, layer = layer)
            output = emb['x'] # [batch, length, dim], the last layer of transformer encoder

            emb = [l[0].transpose(0, 1) for l in emb['layer_results']] # emb[i].shape as same as output.shape
            emb = torch.cat(emb, dim=1).reshape(input_data.shape[0], layer+1, -1, self.out_dim) # list of (bs, frame_num, feat) -> (bs, layer_num, frame_num, feat)

        return output, emb

class Hubert_Base_95m(nn.Module):
    def __init__(self, device, cp_path = os.path.join(_pretrained_model_path, 'hubert_base_ls960.pt')):
        super(Hubert_Base_95m, self).__init__()

        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0]
        self.device = device
        self.out_dim = 768
        self.num_layers = 12
        return

    def extract_feat(self, input_data, layer = 12):

        # put the model to GPU if it not there
        if next(self.model.parameters()).device != input_data.device \
           or next(self.model.parameters()).dtype != input_data.dtype:
            self.model.to(input_data.device, dtype=input_data.dtype)
            # self.model.train()


        if True:
            # input should be in shape (batch, length)
            if input_data.ndim == 3:
                input_tmp = input_data[:, :, 0]
            else:
                input_tmp = input_data

            # Extract features from each layer
            feature, padding_mask, layer_results = self.model.extract_features(input_tmp, padding_mask = None, output_layer = layer)
            output = feature # [batch, length, dim], the last layer of transformer encoder
            # print(output.shape)

            emb = [l[0].transpose(0, 1) for l in layer_results] # emb[i].shape as same as output.shape
            # print(len(emb), emb[0].shape)
            emb = torch.cat(emb, dim=1).reshape(input_data.shape[0], layer, -1, self.out_dim) # list of (bs, frame_num, feat) -> (bs, layer_num, frame_num, feat)

        return output, emb


class Hubert_Large_316m(nn.Module):
    def __init__(self, device, cp_path = os.path.join(_pretrained_model_path, 'hubert_large_ll60k.pt')):
        super(Hubert_Large_316m, self).__init__()

        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0]
        self.device = device
        self.out_dim = 1024
        self.num_layers = 24
        return

    def extract_feat(self, input_data, layer = 24):

        # put the model to GPU if it not there
        if next(self.model.parameters()).device != input_data.device \
           or next(self.model.parameters()).dtype != input_data.dtype:
            self.model.to(input_data.device, dtype=input_data.dtype)
            # self.model.train()


        if True:
            # input should be in shape (batch, length)
            if input_data.ndim == 3:
                input_tmp = input_data[:, :, 0]
            else:
                input_tmp = input_data

            # Extract features from each layer
            feature, padding_mask, layer_results = self.model.extract_features(input_tmp, padding_mask = None, output_layer = layer)
            output = feature # [batch, length, dim], the last layer of transformer encoder

            emb = [l[0].transpose(0, 1) for l in layer_results] # emb[i].shape as same as output.shape
            emb = torch.cat(emb, dim=1).reshape(input_data.shape[0], layer, -1, self.out_dim) # list of (bs, frame_num, feat) -> (bs, layer_num, frame_num, feat)

        return output, emb


def make_sslmodel(class_name):
    if class_name == 'wav2vec2_xlsr_300m':
        return Wav2vec2_XLSR_300m
    elif class_name == 'wav2vec2_xlsr_1b':
        return Wav2vec2_XLSR_1b
    elif class_name == 'hubert_base_95m':
        return Hubert_Base_95m
    elif class_name == 'hubert_large_316m':
        return Hubert_Large_316m
    else:
        raise ValueError("Invalid class name")
