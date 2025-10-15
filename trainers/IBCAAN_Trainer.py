import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.network_utils import Classifier, MLP
from .Base_Trainer import Base_Trainer
from utils import (
    str_to_bool,
    calc_eer_asvspoof,
    get_model_entity,
    create_optimizer
)
from torchcontrib.optim import SWA

import Adver_network


class IBCAAN_Trainer(Base_Trainer):
    def __init__(self, model, device, optim_config, config):
        super().__init__(model, device, optim_config, config)
        self.device = device
        self.config = config
        self.optim_config = optim_config

        self.featurizer = model.Featurizer
        self.classifier = model.Classifier

        feat_dim = self.featurizer.hidden_dim
        out_dim = self.classifier.out_dim
        cls_dim = self.classifier.hidden_dim

        self.n_attacks = config["n_attacks"] # 4 for asvspoof2019, 8 for asvspoof5
        if str_to_bool(config["conditional"]):
            adv_feat_dim = config["latent_dim"] + 1
        else:
            adv_feat_dim = config["latent_dim"]
        self.domain_cls = MLP(adv_feat_dim, self.n_attacks, config["latent_dim"],
                              3, 0.1, str_to_bool(config['enable_bn']))

        # multi-gpu
        self.is_multi_gpu = str_to_bool(config["multi_gpu"])
        device_ids = list(config["device_ids"])

        if self.is_multi_gpu and torch.cuda.device_count() >= len(device_ids):
            self.featurizer = nn.DataParallel(self.featurizer, device_ids=device_ids)
            self.classifier = nn.DataParallel(self.classifier, device_ids=device_ids)
            self.domain_cls = nn.DataParallel(self.domain_cls, device_ids=device_ids)
            print('Enable DataParallel, device {}'.format(device_ids))

        # reparam components
        if str_to_bool(config['enable_reparam']):
            assert config["latent_dim"] == cls_dim
            latent_dim = config["latent_dim"]
            self.encoder = MLP(feat_dim, latent_dim, 4 * latent_dim, 3,
                               0, str_to_bool(config['enable_bn']))
            self.fc_mu = nn.Linear(latent_dim, latent_dim)
            self.fc_logvar = nn.Linear(latent_dim, latent_dim)

        # optimizer
        params = []
        params.extend(self.featurizer.parameters())
        params.extend(self.classifier.parameters())
        if str_to_bool(config['enable_reparam']):
            params.extend(self.encoder.parameters())
            params.extend(self.fc_mu.parameters())
            params.extend(self.fc_logvar.parameters())
        self.optimizer, self.scheduler = create_optimizer(params, optim_config)
        self.optimizer_swa = SWA(self.optimizer)

        self.domain_optimizer, _ = create_optimizer(
            self.domain_cls.parameters(), optim_config)

        # classification loss
        if config['loss'] == 'CCE':
            self.criterion = nn.CrossEntropyLoss()
        elif config['loss'] == 'WCE':
            _weight = torch.FloatTensor([0.1, 0.9]).to(device)
            self.criterion = nn.CrossEntropyLoss(weight=_weight)
        else:
            raise ValueError(f"Unknown Loss: {config['loss']}")

        self.domain_criterion = nn.CrossEntropyLoss()

        self.lambda_ib = config.get('lambda_ib', 0.0)
        self.lambda_adv = self.config.get("lambda_adv", 0.0)

        self.alpha_sheduler = Adver_network.AlphaSheduler(
            gamma=10.0, max_iter=config["num_epochs"] / 2)

        self._real_attack_id = 0

    @property
    def real_attack_id(self):
        return self._real_attack_id

    @real_attack_id.setter
    def real_attack_id(self, value):
        self._real_attack_id = value

    def encoder_fun(self, res_feat):
        latent_z = self.encoder(res_feat)
        mu = self.fc_mu(latent_z)
        logvar = self.fc_logvar(latent_z)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar / 2)
            eps = torch.randn_like(std)
            return mu + std * eps
        else:
            return mu

    def get_features(self, x, stats=False):
        z = self.featurizer(x)
        
        mu, logvar = None, None
        if str_to_bool(self.config['enable_reparam']):
            mu, logvar = self.encoder_fun(z)
            z = self.reparameterize(mu, logvar)
        else:
            z = z

        if stats:
            return z, mu, logvar
        else:
            return z

    def predict(self, x):
        z = self.get_features(x)
        y_hat = self.classifier(z)
        return y_hat

    def forward(self, x):
        return self.predict(x)

    def train_one_epoch(self, trn_loader, epoch, verbose=False):
        running_loss, running_cls_loss, running_ib_loss = 0.0, 0.0, 0.0
        running_adv_loss, running_adv_correct = 0.0, 0.0
        num_total, num_adv_total = 0, 0
        self.train()

        self.alpha_sheduler.step()

        for batch_x, batch_y, batch_speaker, batch_attack in trn_loader:
            batch_size = batch_x.size(0)
            num_total += batch_size

            batch_x = batch_x.to(self.device, non_blocking=True)
            batch_y = batch_y.view(-1).type(torch.int64).to(self.device, non_blocking=True)
            batch_attack = batch_attack.view(-1).type(torch.int64).to(self.device, non_blocking=True)

            bonafide_mask = (batch_y == 1)
            spoof_mask = (batch_y == 0)

            # Extract features from shared extractor
            main_z, mu, logvar = self.get_features(batch_x, stats=True)

            # Information bottleneck for main branch
            ib_loss = torch.tensor(0.0, device=self.device)
            if str_to_bool(self.config['enable_reparam']):
                ib_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size

            # Classification loss for main task
            batch_logit = self.classifier(main_z)
            cls_loss = self.criterion(batch_logit, batch_y)

            # Adversarial Domain Classification
            z_spoof = main_z[spoof_mask]
            attack_spoof = batch_attack[spoof_mask]
            adv_loss = torch.tensor(0.0, device=self.device)
            if z_spoof.size(0) > 0:
                if str_to_bool(self.config["conditional"]):
                    spoof_conf_scores = torch.softmax(
                        batch_logit.detach(), dim=1)[spoof_mask, 0].unsqueeze(1)
                    # spoof_conf_scores = torch.sigmoid(
                    #     (batch_logit[spoof_mask, 0] - batch_logit[spoof_mask, 1]).unsqueeze(1).detach()
                    # )
                    z_spoof = torch.cat([z_spoof, spoof_conf_scores], dim=1)
                adv_feat = Adver_network.ReverseLayerF.apply(z_spoof, self.alpha_sheduler())
                adv_pred = self.domain_cls(adv_feat)
                adv_loss = self.domain_criterion(adv_pred, attack_spoof)
                # accuracy of domain_cls
                with torch.no_grad():
                    _, predicted_adv = torch.max(adv_pred, 1)
                    correct_adv = (predicted_adv == attack_spoof).sum().item()
                    running_adv_correct += correct_adv
                    num_adv_total += attack_spoof.size(0)

            # Total loss
            total_loss = (cls_loss +
                          self.lambda_adv * adv_loss +
                          self.lambda_ib * ib_loss)

            # Backward and optimize
            self.optimizer.zero_grad()
            self.domain_optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            self.domain_optimizer.step()

            # Record losses
            running_loss += total_loss.item() * batch_size
            running_cls_loss += cls_loss.item() * batch_size
            running_ib_loss += ib_loss.item() * batch_size
            running_adv_loss += adv_loss.item() * batch_size

            # Learning rate scheduling
            if self.optim_config["scheduler"] in ["cosine", "keras_decay", "noam"]:
                self.scheduler.step()

        return {
            'total_loss': running_loss / num_total,
            'cls_loss': running_cls_loss / num_total,
            'ib_loss': running_ib_loss / num_total,
            'adv_loss': running_adv_loss / num_total,
            "adv_accuracy": running_adv_correct / num_adv_total
        }

    def save_checkpoint(self, save_path):
        save_dict = {
            "featurizer_dict": get_model_entity(self.featurizer, self.is_multi_gpu).state_dict(),
            "classifier_dict": get_model_entity(self.classifier, self.is_multi_gpu).state_dict(),
        }
        if str_to_bool(self.config['enable_reparam']):
            save_dict["encoder_dict"] = self.encoder.state_dict()
            save_dict["fc_mu_dict"] = self.fc_mu.state_dict()
            save_dict["fc_logvar_dict"] = self.fc_logvar.state_dict()

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

        if str_to_bool(self.config['enable_reparam']):
            self.encoder.load_state_dict(save_dict["encoder_dict"])
            self.fc_mu.load_state_dict(save_dict["fc_mu_dict"])
            self.fc_logvar.load_state_dict(save_dict["fc_logvar_dict"])

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

            for key in ["featurizer_dict", "classifier_dict",
                        "encoder_dict", "fc_mu_dict", "fc_logvar_dict"]:
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
                state_dict[k] = (v / n).to(v.dtype)  # keep float32

            if key == "featurizer_dict":
                self.featurizer.load_state_dict(state_dict)
            elif key == "classifier_dict":
                self.classifier.load_state_dict(state_dict)
            elif key == "encoder_dict" and str_to_bool(self.config['enable_reparam']):
                self.encoder.load_state_dict(state_dict)
            elif key == "fc_mu_dict" and str_to_bool(self.config['enable_reparam']):
                self.fc_mu.load_state_dict(state_dict)
            elif key == "fc_logvar_dict" and str_to_bool(self.config['enable_reparam']):
                self.fc_logvar.load_state_dict(state_dict)

        if save_path is not None:
            torch.save(avg_state_dict, save_path)
            print(f"Averaged checkpoint saved at {save_path}")

        print(f"Averaged {n} checkpoints successfully.")

