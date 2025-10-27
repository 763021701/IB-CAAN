# IB-CAAN: Information Bottleneck enhanced Confidence-Aware Adversarial Network

Official PyTorch implementation of [Generalizable Speech Deepfake Detection via Information Bottleneck Enhanced Adversarial Alignment](https://arxiv.org/abs/2509.23618).

## 🌿 Branch Overview

- **master**: For *ASVspoof 2019 LA*, *ASVspoof 2021 LA and DF* and *In-the-Wild* datasets 👉 [master branch](https://github.com/763021701/IB-CAAN).

- **asvspoof5**: For *ASVspoof 5* dataset.

## 🔥 Update:
✔️ [October 15, 2025] The code for *ASVspoof 2019 LA*, *ASVspoof 2021*, and *In-the-Wild* has been released.

✔️ [October 27, 2025] The code for *ASVspoof 5* has been released.



# Preparation

## Dependencies

Install requirements
```bash
pip install -r requirements.txt
```

Install fairseq
```bash
git clone https://github.com/facebookresearch/fairseq.git
cd fairseq
git checkout a54021305d6b3c
pip install --editable ./
```

## Datasets

Our experiments are performed on [*ASVspoof 5*](https://zenodo.org/records/14498691) and [*In-the-Wild*](https://deepfake-total.com/in_the_wild). Download the above datasets and organize them into the following structure:

```bash
/change/to/your/path/
├── ASVspoof5
│   ├── ASVspoof5_protocols
│   ├── flac_T
│   ├── flac_D
│   └── flac_E
└── release_in_the_wild
```

In some experiments, we apply [*MUSAN noise*](https://www.openslr.org/17/) and [*RIR*](https://www.openslr.org/28/) for data augmentation. Download the above data and modify the path in the configuration file:

```json
{
  "data_aug_config": {
    "aug_list": [],
    "musan_dir": "/change/to/your/path/musan/noise",
    "rir_dir": "/change/to/your/path/RIRS_NOISES/simulated_rirs"
  }
}
```

## Pre-trained Models

Download the [*XLS-R 300M*](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec/xlsr) pre-trained model and modify the path in models/ssl_model.py:

```python
_pretrained_model_path = '/change/to/your/xlsr/path/'
```

# Train & Evaluation

Activate project environment variables:

```bash
source project_env.sh
```

### Train

For example, if you use the configuration file *Wav2vec2_XLSR_ASVspoof5_IBCAAN.conf* for training, you should modify the field "database_path" to point to your paths. And run:

```bash
python main.py --config config/Wav2vec2_XLSR_ASVspoof5_IBCAAN.conf
```

### Evaluation

Modify the following fields in the configuration file:

```json
{
  "model_path": "/change/to/your/model/path/",
  "checkpoints":  ["checkpoint0.pth", "checkpoint1.pth", "checkpoint2.pth"],
}
```

- If the length of **"checkpoints"** >= 1, weighted averaging is automatically performed.


Then execute the following command to generate the evaluation scores:

```bash
python main.py --config config/Wav2vec2_XLSR_ASVspoof5_IBCAAN.conf --eval
```

Get the evaluation results:

```bash
calc_eer_ASV5 /your/ASV5/scores.txt
```

# Main Results

Results are reported as the format of *best/mean* across 3 runs.

| $\Phi_\theta$    | $f_\omega$ | Track1 | Ckpts & Scores |
|:------:|:------:|:------:|:------:|
| RawBMamba | Linear  | 30.59 / 31.01 | TODO. |
| RawBMamba* | Linear*  | 27.61 / 27.77 | TODO. |
| XLSR      | MLP    | 5.96 / 5.98 | TODO. |
| XLSR*      | MLP*    | 4.38 / 4.67 | [Link](https://drive.google.com/drive/folders/1xXhk1Kf1uOBCSw221xc-8b8fyzSMFWZk?usp=drive_link) |

\* with Musan noise and RIR augmentations.


# Reference Repo
Thanks for following open-source projects:                                                                  

1. Rawboost: https://github.com/TakHemlata/RawBoost-antispoofing Paper: [[Rawboost]](https://arxiv.org/abs/2111.04433)
2. Nes2Net: https://github.com/Liu-Tianchi/Nes2Net_ASVspoof_ITW Paper: [[Nes2Net]](https://arxiv.org/abs/2504.05657)
3. RawBMamba: https://github.com/cyjie429/RawBMamba Paper: [[RawBMamba]](https://arxiv.org/abs/2406.06086)
4. Asvspoof 5 Baselines: https://github.com/asvspoof-challenge/asvspoof5 Paper: [[Asvspoof5]](https://arxiv.org/abs/2408.08739)
5. Transfer Learning: https://github.com/jindongwang/transferlearning Paper: [[DG Survey]](https://arxiv.org/abs/2103.03097)

# Citation

If you use this codebase, or otherwise find our work valuable, please cite:
```
@article{huang2025ibcaan,
  title={Generalizable Speech Deepfake Detection via Information Bottleneck Enhanced Adversarial Alignment},
  author={Pu Huang and Shouguang Wang and Siya Yao and Mengchu Zhou},
  journal={arXiv preprint arXiv:2509.23618},
  year={2025}
}
```