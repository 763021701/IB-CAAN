import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .Base_Trainer import Base_Trainer
from utils import (
    str_to_bool,
    calc_eer_asvspoof,
    get_model_entity,
    create_optimizer
)
from tqdm import tqdm
from torchcontrib.optim import SWA


class ERM_Trainer(Base_Trainer):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, model, device, optim_config, config):
        super().__init__(model, device, optim_config, config)
        self.device = device
        self.config = config
        self.optim_config = optim_config

        self.featurizer = model.Featurizer
        self.classifier = model.Classifier
        self.network = model

        self.is_multi_gpu = str_to_bool(config["multi_gpu"])
        device_ids = list(config["device_ids"])

        if self.is_multi_gpu and torch.cuda.device_count() >= len(device_ids):
            self.featurizer = nn.DataParallel(self.featurizer, device_ids=device_ids)
            self.classifier = nn.DataParallel(self.classifier, device_ids=device_ids)
            self.network = nn.DataParallel(self.network, device_ids=device_ids)
            print('Enable DataParallel, device {}'.format(device_ids))

        params = []
        params.extend(self.featurizer.parameters())
        params.extend(self.classifier.parameters())
        self.optimizer, self.scheduler = create_optimizer(params, optim_config)
        self.optimizer_swa = SWA(self.optimizer)

        if config['loss'] == 'CCE':
            self.criterion = nn.CrossEntropyLoss()
        elif config['loss'] == 'WCE':
            _weight = torch.FloatTensor([0.1, 0.9]).to(device)
            self.criterion = nn.CrossEntropyLoss(weight=_weight)
        else:
            raise ValueError(f"Unknown Los: {config['loss']}")

    def get_features(self, x):
        z = self.featurizer(x)
        return z

    def predict(self, x):
        z = self.featurizer(x)
        y_hat = self.classifier(z)
        return y_hat

    def forward(self, x):
        return self.predict(x)

    def train_one_epoch(self, trn_loader, epoch, verbose=False):
        running_loss = 0.0
        running_cls_loss = 0.0
        num_total = 0
        self.train()

        for batch_x, batch_y, _, batch_attack in trn_loader:
            batch_size = batch_x.size(0)
            num_total += batch_size

            batch_x = batch_x.to(self.device, non_blocking=True)
            batch_y = batch_y.view(-1).type(torch.int64).to(self.device, non_blocking=True)
            batch_attack = batch_attack.view(-1).type(torch.int64).to(self.device, non_blocking=True)

            # Forward pass
            batch_feature = self.featurizer(batch_x)
            batch_logit = self.classifier(batch_feature)

            # Classification loss
            cls_loss = self.criterion(batch_logit, batch_y)

            total_loss = cls_loss
            running_loss += total_loss.item() * batch_size
            running_cls_loss += cls_loss.item() * batch_size

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            if self.config["optim_config"]["scheduler"] in ["cosine", "keras_decay", "noam"]:
                self.scheduler.step()
            elif self.scheduler is None:
                pass
            else:
                raise ValueError("scheduler error, got:{}".format(self.scheduler))

        return {'loss': running_loss / num_total,
                'cls_loss': running_cls_loss / num_total}

    def save_checkpoint(self, save_path):
        save_dict = {
            "featurizer_dict": get_model_entity(self.featurizer, self.is_multi_gpu).state_dict(),
            "classifier_dict": get_model_entity(self.classifier, self.is_multi_gpu).state_dict(),
        }
        if self.optimizer is not None:
            save_dict["optimizer_dict"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            save_dict["scheduler_dict"] = self.scheduler.state_dict()
        torch.save(save_dict, save_path)

    def load_checkpoint(self, path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        save_dict = torch.load(path, map_location=self.device)

        self.featurizer.load_state_dict(save_dict["featurizer_dict"])
        self.classifier.load_state_dict(save_dict["classifier_dict"])

        if self.optimizer is not None and "optimizer_dict" in save_dict:
            self.optimizer.load_state_dict(save_dict["optimizer_dict"])
        if self.scheduler is not None and "scheduler_dict" in save_dict:
            self.scheduler.load_state_dict(save_dict["scheduler_dict"])

    def load_average_checkpoint(self, paths, save_path=None):
        """
        Load averaged parameters from multiple checkpoints.

        Args:
            paths (list[str]): list of checkpoint file paths to average
            save_path (str, optional): if given, save the averaged checkpoint here
        """
        if not isinstance(paths, (list, tuple)):
            raise ValueError("paths must be a list of checkpoint file paths")

        avg_state_dict = {}
        n = len(paths)

        for i, path in enumerate(paths):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Checkpoint file not found: {path}")
            save_dict = torch.load(path, map_location=self.device)

            for key in ["featurizer_dict", "classifier_dict"]:
                if key not in save_dict:
                    continue

                state_dict = save_dict[key]

                if key not in avg_state_dict:
                    avg_state_dict[key] = {k: v.clone().to(torch.float32) for k, v in state_dict.items()}
                else:
                    for k in state_dict:
                        avg_state_dict[key][k] += state_dict[k].to(torch.float32)

        for key, state_dict in avg_state_dict.items():
            for k, v in state_dict.items():
                state_dict[k] = (v / n).to(v.dtype)

            if key == "featurizer_dict":
                self.featurizer.load_state_dict(state_dict)
            elif key == "classifier_dict":
                self.classifier.load_state_dict(state_dict)

        if save_path is not None:
            torch.save(avg_state_dict, save_path)
            print(f"Averaged checkpoint saved at {save_path}")

        print(f"Averaged {n} checkpoints successfully.")