"""
Utilization functions
"""

import os
import random
import sys

import numpy as np
import torch

from evaluation import calculate_tDCF_EER

from collections import defaultdict

def str_to_bool(val):
    """Convert a string representation of truth to true (1) or false (0).
    Copied from the python implementation distutils.utils.strtobool

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    >>> str_to_bool('YES')
    1
    >>> str_to_bool('FALSE')
    0
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    if val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    raise ValueError('invalid truth value {}'.format(val))


def cosine_annealing(step, total_steps, lr_max, lr_min):
    """Cosine Annealing for learning rate decay scheduler"""
    return lr_min + (lr_max -
                     lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


def keras_decay(step, decay=0.0001):
    """Learning rate decay in Keras-style"""
    return 1. / (1. + decay * step)


class SGDRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """SGD with restarts scheduler"""
    def __init__(self, optimizer, T0, T_mul, eta_min, last_epoch=-1):
        self.Ti = T0
        self.T_mul = T_mul
        self.eta_min = eta_min

        self.last_restart = 0

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        T_cur = self.last_epoch - self.last_restart
        if T_cur >= self.Ti:
            self.last_restart = self.last_epoch
            self.Ti = self.Ti * self.T_mul
            T_cur = 0

        return [
            self.eta_min + (base_lr - self.eta_min) *
            (1 + np.cos(np.pi * T_cur / self.Ti)) / 2
            for base_lr in self.base_lrs
        ]


def _get_optimizer(model_parameters, optim_config):
    """Defines optimizer according to the given config"""
    optimizer_name = optim_config['optimizer']

    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model_parameters,
                                    lr=optim_config['base_lr'],
                                    momentum=optim_config['momentum'],
                                    weight_decay=optim_config['weight_decay'],
                                    nesterov=optim_config['nesterov'])
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model_parameters,
                                     lr=optim_config['base_lr'],
                                     betas=optim_config['betas'],
                                     weight_decay=optim_config['weight_decay'],
                                     amsgrad=str_to_bool(
                                         optim_config['amsgrad']))
    else:
        print('Un-known optimizer', optimizer_name)
        sys.exit()

    return optimizer


def _get_scheduler(optimizer, optim_config):
    """
    Defines learning rate scheduler according to the given config
    """
    if optim_config['scheduler'] == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=optim_config['milestones'],
            gamma=optim_config['lr_decay'])

    elif optim_config['scheduler'] == 'sgdr':
        scheduler = SGDRScheduler(optimizer, optim_config['T0'],
                                  optim_config['Tmult'],
                                  optim_config['lr_min'])

    elif optim_config['scheduler'] == 'cosine':
        total_steps = optim_config['epochs'] * \
            optim_config['steps_per_epoch']

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                total_steps,
                1,  # since lr_lambda computes multiplicative factor
                optim_config['lr_min'] / optim_config['base_lr']))

    elif optim_config['scheduler'] == 'keras_decay':
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: keras_decay(step))

    elif optim_config['scheduler'] == 'noam':
        d_model = optim_config['d_model']
        warmup  = optim_config['warmup_steps']
        def noam_lambda(step):
            step += 1   # 1-based
            scale = d_model ** -0.5
            lr = scale * min(step ** -0.5, step * warmup ** -1.5)
            return lr / optim_config['base_lr']   # LambdaLR 会把返回值乘到 base_lr

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=noam_lambda)
    else:
        scheduler = None
    return scheduler


def create_optimizer(model_parameters, optim_config):
    """Defines an optimizer and a scheduler"""
    optimizer = _get_optimizer(model_parameters, optim_config)
    scheduler = _get_scheduler(optimizer, optim_config)
    return optimizer, scheduler


def seed_worker(worker_id):
    """
    Used in generating seed for the worker of torch.utils.data.Dataloader
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(seed, config = None):
    """ 
    set initial seed for reproduction
    """
    if config is None:
        raise ValueError("config should not be None")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = str_to_bool(config["cudnn_deterministic_toggle"])
        torch.backends.cudnn.benchmark = str_to_bool(config["cudnn_benchmark_toggle"])


def mixup_process(out, target_reweighted, lam):
    indices = np.random.permutation(out.size(0))
    out = out*lam + out[indices]*(1-lam)
    target_shuffled_onehot = target_reweighted[indices]
    target_reweighted = target_reweighted * lam + target_shuffled_onehot * (1 - lam)
    return out, target_reweighted


def get_lambda(alpha=1.0):
    '''Return lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    return lam


def print_dict(**kwargs):
    """
    use: print_result(**result)
    """
    for k, v in kwargs.items():
        print(f'{k}: {v:.5f}', end=' ')
    print()


def write_log_dict(log_file, **kwargs):
    """
    use: print_result(**result)
    """
    for k, v in kwargs.items():
        log_file.write(f'{k}: {v:.5f}\n')


def save_checkpoint(trainer, filename):
    save_dict = {
        "trainer_dict": trainer.model.cpu().state_dict()
    }
    torch.save(save_dict, filename)


def calc_eer_asvspoof(track, metric_path, database_path, fname_list, score_list, config):
    """
    track: 'asvspoof2019' 'asvspoof5_split' 'asvspoof5_dev' 'asvspoof5_eval'
    """
    score_path = metric_path / "dev_score.txt"
    output_file = metric_path / "dev_t-DCF_EER_ep_{}.txt".format(track)

    if track.startswith('asvspoof2019'):
        asv_score_file = database_path / "ASVspoof2019_LA" / config["asv_score_path"]
        trial_path = (database_path / "ASVspoof2019_LA" /
                      "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt")

        with open(trial_path, "r") as f_trl:
            trial_lines = f_trl.readlines()
        assert len(trial_lines) == len(fname_list) == len(score_list)

        with open(score_path, "w") as fh:
            for fn, sco, trl in zip(fname_list, score_list, trial_lines):
                _, utt_id, _, src, key = trl.strip().split(' ')
                assert fn == utt_id
                fh.write("{} {} {} {}\n".format(utt_id, src, key, sco))

        dev_eer, dev_tdcf = calculate_tDCF_EER(
            cm_scores_file=score_path,
            asv_score_file=asv_score_file,
            output_file=output_file,
            printout=False)
        
        return dev_eer, dev_tdcf, 0
    
    else:
        raise ValueError(f"Unknown Track: {track}")

def get_model_entity(model, multi_gpu):
    if multi_gpu:
        return model.module
    else:
        return model