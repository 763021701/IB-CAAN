import torch
import torch.nn as nn
import torch.nn.functional as F

from .ssl_model import make_sslmodel
from .network_utils import Classifier, ASTP
from utils import str_to_bool


class Wav2vec2_XLSR_Featurizer(nn.Module):
     def __init__(self, d_args):
         super(Wav2vec2_XLSR_Featurizer, self).__init__()
         self.device = d_args["device"]
         self.hidden_dim = d_args["emb_dim"]
         self.frame_pool = d_args["frame_pool"]

         # SSLModel
         self.ssl_model = make_sslmodel('wav2vec2_xlsr_300m')(self.device)
         self.proj = nn.Linear(self.ssl_model.out_dim, self.hidden_dim)

         # Frame-level pooling
         if self.frame_pool == 'mean':
             self.pool = nn.AdaptiveAvgPool1d(1)
             self.post_proj = nn.Identity()
         elif self.frame_pool == 'astp':
             self.pool = ASTP(self.hidden_dim, self.hidden_dim // 2, False)
             self.post_proj = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
         else:
             raise ValueError("Invalid frame pooling")

     def forward(self, x):
         x, _ = self.ssl_model.extract_feat(x) # (B,T,F) (batch_size, time_frame, feat_dim)
         x = self.proj(x)  # (B,T,D) (batch_size, time_frame, hidden_dim)
         x = x.transpose(1, 2) # (B, D, T)
         x = self.pool(x) # (B, D, 1) 'mean', (B, 2*D, 1) 'astp'
         x = x.squeeze(-1)
         x = self.post_proj(x) # (B, D)
         return x


class Wav2vec2_XLSR_Classifier(nn.Module):
    def __init__(self, d_args):
        super().__init__()
        self.d_args = d_args
        self.hidden_dim = d_args["emb_dim"]
        self.out_dim = 2

        self.out_layer = Classifier(
            in_dim = self.hidden_dim,
            out_dim = self.out_dim,
            is_nonlinear = str_to_bool(d_args["nonlinear_classifier"]))

    def forward(self, x):
        output = self.out_layer(x)
        return output


class Model(nn.Module):
    def __init__(self, d_args):
        super().__init__()
        self.Featurizer = Wav2vec2_XLSR_Featurizer(d_args)
        self.Classifier = Wav2vec2_XLSR_Classifier(d_args)

    def forward(self, x):
        hidden = self.Featurizer(x)
        output = self.Classifier(hidden)
        return hidden, output
