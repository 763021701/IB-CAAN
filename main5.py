"""
Main script that trains, validates, and evaluates
various models including AASIST.

AASIST
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import argparse
import json
import os
import sys
import datetime
import warnings
from importlib import import_module
from pathlib import Path
from shutil import copy
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from data_utils5 import (
    genSpoof_list,
    Dataset_ASVspoof5_train,
    Dataset_ASVspoof5_train_for_SWA,
    Dataset_ASVspoof5_dev,
    Dataset_ASVspoof5_eval,
    ATTACK_2_INT
)

from utils import (
    seed_worker,
    set_seed,
    str_to_bool,
    print_dict,
    write_log_dict
)

warnings.filterwarnings("ignore", category=FutureWarning)


def main(args: argparse.Namespace) -> None:
    """
    Main function.
    Trains, validates, and evaluates the ASVspoof detection model.
    """
    # load experiment configurations
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())
    model_config = config["model_config"]
    model_config["device"] = config["device"]
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]

    # make experiment reproducible
    set_seed(args.seed, config)

    # ckpt path
    if len(config["checkpoints"]) > 1:
        model_path = os.path.join(config["model_path"], "model_avg.pth")
        ckpt_paths = []
        for p in config["checkpoints"]:
            ckpt_paths.append(os.path.join(config["model_path"], p))
    elif len(config["checkpoints"]) == 1:
        model_path = os.path.join(config["model_path"], config["checkpoints"][0])
        ckpt_paths = os.path.join(config["model_path"], config["checkpoints"][0])
    else:
        raise ValueError("Invalid checkpoints")
    print(model_path)
    print(ckpt_paths)

    # define database related paths
    output_dir = Path(args.output_dir)
    database_path = Path(config["database_path"])
    # define model related paths
    model_tag = "{}_ep{}_bs{}".format(
        os.path.splitext(os.path.basename(args.config))[0],
        config["num_epochs"], config["batch_size"])
    if args.comment:
        model_tag = model_tag + "_{}".format(args.comment)
    model_tag = output_dir / model_tag
    model_save_path = model_tag / "weights"
    eval_score_path = (model_tag / "eval_scores_{}_{}.txt".format(
        os.path.basename(model_path).replace('model_', '').replace('.pth', ''),
        config["track"]))
    writer = SummaryWriter(model_tag)
    os.makedirs(model_save_path, exist_ok=True)
    copy(args.config, model_tag / "config.conf")

    # set device
    device = config["device"] if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))
    if device == "cpu":
        raise ValueError("GPU not detected!")

    # define model architecture
    model = get_model(model_config, device)

    # define dataloaders
    trn_loader, _, dev_loader = get_loader(database_path, args.seed, config)
    # define online testing dataloaders
    # dev_track1_loader, eval_track1_loader = get_online_eval_loader(database_path, config)

    optim_config["steps_per_epoch"] = len(trn_loader)
    # get trainer
    trainer = get_trainer(model, device, optim_config, config)
    trainer.to(device)

    # evaluates pretrained model
    if args.eval:
        if len(config["checkpoints"]) == 1:
            trainer.load_checkpoint(model_path)
        else:
            trainer.load_average_checkpoint(ckpt_paths, model_path)
        print("Model loaded : {}".format(model_path))

        eval_loader = get_eval_loader(database_path, config)
        print("Dataloader in track {}".format(config["track"]))

        print("Start evaluation...")
        trial_path = database_path / "ASVspoof5_protocols/ASVspoof5.eval.track_1.tsv"
        trainer.produce_evaluation_file(eval_loader, trial_path, eval_score_path)
        sys.exit(0)

    val_mode = config.get("val_mode", "loss")
    top5_checkpoints = []  # Each (-metric, epoch)
    f_log = open(model_tag / "metric_log.txt", "a")

    # make directory for metric logging
    metric_path = model_tag / "metrics"
    os.makedirs(metric_path, exist_ok=True)

    # Training
    for epoch in range(config["num_epochs"]):
        train_result = trainer.train_one_epoch(trn_loader, epoch)
        val_result = trainer.validation_one_epoch(dev_loader, metric_path, database_path, 'asvspoof5_dev')

        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("\n[{}]".format(current_time))

        merged_result = {'epoch': epoch,
                         **{f't_{k}': v for k, v in train_result.items()},
                         **{f'v_{k}': v for k, v in val_result.items()}}

        print_dict(**merged_result)
        write_log_dict(f_log, **merged_result)
        for k, v in merged_result.items():
            writer.add_scalar(k, v, epoch)

        current_metric = val_result['val_loss'] if val_mode == "loss" else val_result['dev_eer']
        top5_checkpoints.append((current_metric, epoch))
        top5_checkpoints.sort(key=lambda x: x[0])
        if len(top5_checkpoints) > 5:
            top5_checkpoints.pop()

        trainer.save_checkpoint(model_save_path / "model_epoch_{}.pth".format(epoch))

    print("\n" + "="*50)
    print("Final Top-5 Checkpoints:")
    for i, (metric_val, ep) in enumerate(top5_checkpoints, 1):
        print(f"  Epoch {ep}: {val_mode} = {metric_val:.6f}")
        f_log.write(f"  Epoch {ep}: {val_mode} = {metric_val:.6f}")


def get_model(model_config: Dict, device: torch.device):
    """Define DNN model architecture"""
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))

    return model


def get_trainer(model, device, optim_config, config):
    if config["trainer"] == "erm":
        from trainers.ERM_Trainer import ERM_Trainer
        return ERM_Trainer(model, device, optim_config, config)
    elif config["trainer"] == "ibcaan":
        from trainers.IBCAAN_Trainer import IBCAAN_Trainer
        trainer = IBCAAN_Trainer(model, device, optim_config, config)
        trainer.real_attack_id = ATTACK_2_INT['bonafide']
        return trainer
    else:
        raise ValueError("Unknown trainer: {}".format(config["trainer"]))


def get_loader(
    database_path: str,
    seed: int,
    config: dict) -> List[torch.utils.data.DataLoader]:
    """Make PyTorch DataLoaders for train / developement"""

    trn_database_path = database_path / "flac_T/"
    dev_database_path = database_path / "flac_D/"

    trn_list_path = (database_path /
                     "ASVspoof5_protocols/ASVspoof5.train.tsv")
    dev_trial_path = (database_path /
                      "ASVspoof5_protocols/ASVspoof5.dev.track_1.tsv")

    d_label_trn, d_attack_trn, file_train = genSpoof_list(
        dir_meta=trn_list_path, is_train=True, is_eval=False)
    print("no. training files:", len(file_train))

    train_set = Dataset_ASVspoof5_train(
        config["data_aug_config"],
        list_IDs=file_train,
        labels=d_label_trn,
        base_dir=trn_database_path,
        attacks=d_attack_trn
    )
    gen = torch.Generator()
    gen.manual_seed(seed)
    trn_loader = DataLoader(
        train_set,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=gen,
        num_workers=config["num_workers"]
    )

    train_set_swa = Dataset_ASVspoof5_train_for_SWA(
        config["data_aug_config"],
        list_IDs=file_train,
        labels=d_label_trn,
        base_dir=trn_database_path,
    )
    trn_loader_swa = DataLoader(
        train_set_swa,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=gen,
        num_workers=config["num_workers"]
    )

    d_label_dev, d_attack_dev, file_dev = genSpoof_list(
        dir_meta=dev_trial_path, is_train=False, is_eval=False)
    print("no. validation files:", len(file_dev))

    dev_set = Dataset_ASVspoof5_dev(
        config["data_aug_config"],
        list_IDs=file_dev,
        labels=d_label_dev,
        base_dir=dev_database_path,
        attacks=d_attack_dev
    )
    dev_loader = DataLoader(
        dev_set,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=config["num_workers"]
    )

    return trn_loader, trn_loader_swa, dev_loader


def get_online_eval_loader(
    database_path: str,
    config: dict) -> List[torch.utils.data.DataLoader]:
    """Make PyTorch DataLoaders for online testing"""

    file_dev = []
    dev_database_path = database_path / "flac_D/"
    dev_trials_path = (database_path /
                   "ASVspoof5_protocols/ASVspoof5.dev.track_1.tsv")
    file_dev = genSpoof_list(dir_meta=dev_trials_path,
                              is_train=False,
                              is_eval=True)
    print("no. dev_track1 files:", len(file_dev))

    dev_set = Dataset_ASVspoof5_eval(list_IDs=file_dev,
                                     base_dir=dev_database_path)
    dev_loader = DataLoader(dev_set,
                            batch_size=config["batch_size"],
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=config["num_workers"])
    
    file_eval = []
    eval_database_path = database_path / "flac_E/"
    eval_trials_path = (database_path /
                        "ASVspoof5_protocols/ASVspoof5.eval.track_1.tsv")
    file_eval = genSpoof_list(dir_meta=eval_trials_path,
                              is_train=False,
                              is_eval=True)
    print("no. eval_track1 files:", len(file_eval))

    eval_set = Dataset_ASVspoof5_eval(list_IDs=file_eval,
                                     base_dir=eval_database_path)

    eval_loader = DataLoader(eval_set,
                            batch_size=config["batch_size"],
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=config["num_workers"])

    return dev_loader, eval_loader

def get_eval_loader(
    database_path: str,
    config: dict) -> List[torch.utils.data.DataLoader]:

    _, eval_loader = get_online_eval_loader(database_path, config)

    return eval_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASVspoof detection system")
    parser.add_argument("--config",
                        dest="config",
                        type=str,
                        help="configuration file",
                        required=True)
    parser.add_argument("--output_dir",
                        dest="output_dir",
                        type=str,
                        help="output directory for results",
                        default="./exp_result")
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="random seed (default: 1234)")
    parser.add_argument("--eval",
                        action="store_true",
                        help="when this flag is given, evaluates given model and exit")
    parser.add_argument("--comment",
                        type=str,
                        default=None,
                        help="comment to describe the saved model")
    main(parser.parse_args())
