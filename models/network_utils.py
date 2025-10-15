import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs, mlp_width, mlp_depth, dropout=0, batch_norm=False):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, mlp_width)
        self.dropout = nn.Dropout(dropout)
        self.hiddens = nn.ModuleList([
            nn.Linear(mlp_width, mlp_width)
            for _ in range(mlp_depth-2)])
        self.output = nn.Linear(mlp_width, n_outputs)
        self.bn = nn.BatchNorm1d(mlp_width) if batch_norm else nn.Identity()
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = self.bn(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = self.bn(x)
            x = F.relu(x)
        x = self.output(x)
        return x


class SNN_MLP(nn.Module):
    """Self-Normalizing Neural Network (SNN) with SELU, Alpha Dropout, and LeCun Init"""
    def __init__(self, n_inputs, n_outputs, mlp_width, mlp_depth, dropout=0.05, use_alpha_dropout=True):
        super(SNN_MLP, self).__init__()
        self.input = nn.Linear(n_inputs, mlp_width)
        if use_alpha_dropout:
            self.dropout = nn.AlphaDropout(dropout)
        else:
            self.dropout = nn.Dropout(dropout)

        self.hiddens = nn.ModuleList([
            nn.Linear(mlp_width, mlp_width)
            for _ in range(mlp_depth - 2)
        ])
        self.output = nn.Linear(mlp_width, n_outputs)
        self.n_outputs = n_outputs

        # === LeCun Normal Initialization using kaiming_normal_ ===
        nn.init.kaiming_normal_(self.input.weight, mode='fan_in', nonlinearity='linear')
        if self.input.bias is not None:
            nn.init.zeros_(self.input.bias)

        for hidden in self.hiddens:
            nn.init.kaiming_normal_(hidden.weight, mode='fan_in', nonlinearity='linear')
            if hidden.bias is not None:
                nn.init.zeros_(hidden.bias)

        nn.init.kaiming_normal_(self.output.weight, mode='fan_in', nonlinearity='linear')
        if self.output.bias is not None:
            nn.init.zeros_(self.output.bias)

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.selu(x)

        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.selu(x)

        x = self.output(x)
        return x

def Classifier(in_dim, out_dim, is_nonlinear=False):
    if is_nonlinear:
        return SNN_MLP(in_dim, out_dim, in_dim, 3, 0.1, True)
    else:
        return torch.nn.Linear(in_dim, out_dim)


class ASTP(nn.Module):
    """ Attentive statistics pooling: Channel- and context-dependent
        statistics pooling, first used in ECAPA_TDNN.
    """

    def __init__(self, in_dim, bottleneck_dim=128, global_context_att=False):
        super(ASTP, self).__init__()
        self.global_context_att = global_context_att

        # Use Conv1d with stride == 1 rather than Linear, then we don't
        # need to transpose inputs.
        if global_context_att:
            self.linear1 = nn.Conv1d(
                in_dim * 3, bottleneck_dim,
                kernel_size=1)  # equals W and b in the paper
        else:
            self.linear1 = nn.Conv1d(
                in_dim, bottleneck_dim,
                kernel_size=1)  # equals W and b in the paper
        self.linear2 = nn.Conv1d(bottleneck_dim, in_dim,
                                 kernel_size=1)  # equals V and k in the paper

    def forward(self, x):
        """
        x: a 3-dimensional tensor in tdnn-based architecture (B,F,T)
            or a 4-dimensional tensor in resnet architecture (B,C,F,T)
            0-dim: batch-dimension, last-dim: time-dimension (frame-dimension)
        """
        if len(x.shape) == 4:
            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        assert len(x.shape) == 3

        if self.global_context_att:
            context_mean = torch.mean(x, dim=-1, keepdim=True).expand_as(x)
            context_std = torch.sqrt(
                torch.var(x, dim=-1, keepdim=True) + 1e-10).expand_as(x)
            x_in = torch.cat((x, context_mean, context_std), dim=1)
        else:
            x_in = x

        # DON'T use ReLU here! ReLU may be hard to converge.
        alpha = torch.tanh(
            self.linear1(x_in))  # alpha = F.relu(self.linear1(x_in))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        var = torch.sum(alpha * (x ** 2), dim=2) - mean ** 2
        std = torch.sqrt(var.clamp(min=1e-10))
        return torch.cat([mean, std], dim=1)