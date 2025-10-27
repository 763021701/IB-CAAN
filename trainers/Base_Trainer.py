import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from utils import (
    str_to_bool,
    calc_eer_asvspoof,
    get_model_entity,
    create_optimizer
)

class Base_Trainer(nn.Module):
    """
    Base Trainer
    """

    def __init__(self, model, device, optim_config, config):
        super().__init__()
        self.config = config
        self.device = device

        self.criterion = None

    def predict(self, x):
        raise NotImplementedError
    
    def forward(self, x):
        return self.predict(x)

    def train_one_epoch(self, verbose=False):
        raise NotImplementedError

    def validation_one_epoch(self, val_loader, metric_path,
                             database_path, track='asvspoof5_dev', verbose=False):
        """Perform validation"""
        val_loss = 0.0
        num_total = 0
        self.eval()

        fname_list = []
        score_list = []
        with torch.no_grad():
            for batch_x, batch_y, utt_id in val_loader:
                batch_size = batch_x.size(0)
                num_total += batch_size
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.view(-1).type(torch.int64).to(self.device, non_blocking=True)

                batch_out = self.predict(batch_x)
                batch_loss = self.criterion(batch_out, batch_y)

                val_loss += batch_loss.item() * batch_size

                batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
                fname_list.extend(utt_id)
                score_list.extend(batch_score.tolist())

        dev_eer, dev_tdcf, _ = calc_eer_asvspoof(
            track, metric_path, database_path, fname_list, score_list, self.config)

        return {'val_loss': val_loss / num_total,
                'dev_eer': dev_eer,
                'dev_tdcf': dev_tdcf}
    
    def produce_evaluation_file(self, eval_loader, trial_path, save_path):
        self.eval()

        fname_list = []
        score_list = []
        for batch_x, utt_id in tqdm(eval_loader):
            batch_x = batch_x.to(self.device, non_blocking=True)
            with torch.no_grad():
                batch_out = self.predict(batch_x)
                batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
            # add outputs
            fname_list.extend(utt_id)
            score_list.extend(batch_score.tolist())

        with open(trial_path, "r") as f_trl:
            trial_lines = f_trl.readlines()
        assert len(trial_lines) == len(fname_list) == len(score_list)

        with open(save_path, "w") as fh:
            for fn, sco, trl in zip(fname_list, score_list, trial_lines):
                spk_id, utt_id, _, _, _, _, _, src, key, _ = trl.strip().split(' ')
                assert fn == utt_id
                fh.write("{} {} {} {}\n".format(spk_id, utt_id, sco, key))
        fh.close()
        print("Scores saved to {}".format(save_path))

    def online_test(self, eval_loader, metric_path,
                    database_path, track='asvspoof5', verbose=False):
        """Perform online testing"""
        self.eval()

        fname_list = []
        score_list = []
        with torch.no_grad():
            for batch_x, utt_id in eval_loader:
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_out = self.predict(batch_x)
                batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
                # add outputs
                fname_list.extend(utt_id)
                score_list.extend(batch_score.tolist())

        eval_eer, eval_tdcf, _ = calc_eer_asvspoof(
            track, metric_path, database_path, fname_list, score_list, self.config)

        return {'eer_{}'.format(track): eval_eer,
                'tdcf_{}'.format(track): eval_tdcf}