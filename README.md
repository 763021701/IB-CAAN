# IB-CAAN: Information Bottleneck enhanced Confidence-Aware Adversarial Network

Official PyTorch implementation of [Generalizable Speech Deepfake Detection via Information Bottleneck Enhanced Adversarial Alignment](https://arxiv.org/abs/2509.23618).

## 🌿 Branch Overview

- **master**: For *ASVspoof 2019 LA*, *ASVspoof 2021 LA and DF* and *In-the-Wild* datasets.

- **asvspoof5**: For *ASVspoof 5* dataset 👉 [asvspoof5 branch](https://github.com/763021701/IB-CAAN/tree/asvspoof5).

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

Our experiments are performed on [*ASVspoof 2019 LA*](https://zenodo.org/records/6906306), [*ASVspoof 2021 LA*](https://zenodo.org/records/4837263), [*ASVspoof 2021 DF*](https://zenodo.org/records/4835108), [*In-the-Wild*](https://deepfake-total.com/in_the_wild), and [*ASVspoof 5*](https://zenodo.org/records/14498691). Download the above datasets and organize them into the following structure:

```bash
/change/to/your/path/
├── ASVspoof2019_LA
│   ├── ASVspoof2019_LA_asv_scores
│   ├── ASVspoof2019_LA_cm_protocols
│   ├── ASVspoof2019_LA_train
│   ├── ASVspoof2019_LA_dev
│   └── ASVspoof2019_LA_eval
├── ASVspoof2021_DF_eval
│   ├── ASVspoof2021_DF_cm_protocols
│   │   └── ASVspoof2021.DF.cm.eval.trl.txt
│   └── flac
├── ASVspoof2021_LA_eval
│   ├── ASVspoof2021_LA_cm_protocols
│   │   └── ASVspoof2021.LA.cm.eval.trl.txt
│   └── flac
├── ASVspoof5
│   ├── ASVspoof5_protocols
│   ├── flac_T
│   ├── flac_D
│   └── flac_E
└── release_in_the_wild
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

For example, if you use the configuration file *Wav2vec2_XLSR_ASVspoof2019_IBCAAN.conf* for training, you should modify the fields "database_path" and "asv_score_path" to point to your paths. And run:

```bash
python main.py --config config/Wav2vec2_XLSR_ASVspoof2019_IBCAAN.conf
```

### Evaluation

Modify the following fields in the configuration file:

```json
{
  "model_path": "/change/to/your/model/path/",
  "checkpoints":  ["checkpoint0.pth", "checkpoint1.pth", "checkpoint2.pth"],
  "track": "19LA",
}
```

- If the length of **"checkpoints"** >= 1, weighted averaging is automatically performed.

- **"track"** can be "19LA", "21LA", "21DF", "ITW".



Then execute the following command to generate the evaluation scores:

```bash
python main.py --config config/Wav2vec2_XLSR_ASVspoof2019_IBCAAN.conf --eval
```

Get the evaluation results:

```bash
calc_eer_19LA /your/19LA/scores/.txt
or
calc_eer_21LA /your/21LA/scores/.txt
or
calc_eer_21DF /your/21DF/scores/.txt
or
calc_eer_ITW /your/ITW/scores/.txt
```

# Main Results

Results are reported as the format of *best/mean* across 3 runs.

| $\Phi_\theta$    | $f_\omega$ | 19LA | 21LA | 21DF | ITW | Ckpts & Scores |
|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| RawBMamba | Linear  | 1.71 / 2.24    | 5.01 / 5.43 | 19.43 / 19.90 | 24.60 / 27.85 | [Link](https://drive.google.com/drive/folders/1q7vbO0w7Szko8HYzvHKak7gwDW1YY5PU?usp=sharing) |
| XLSR      | Linear | 0.37 / 0.58  | 4.66 / 5.06 | 3.28 / 3.51 | 5.54 / 5.99 | [Link](https://drive.google.com/drive/folders/13nR5n5adU6OOXL0Y1R7GiMyf61E7QdtP?usp=sharing) |
| XLSR      | MLP    | 0.24 / 0.40  | 4.00 / 4.69 | 3.50 / 3.75 | 4.61 / 4.93 | [Link](https://drive.google.com/drive/folders/1hbtSt34fvSSxisPywLesi2wN64OcrvHK?usp=sharing) |
| XLSR*      | MLP*    | 0.20 / 0.31  | 2.17 / 2.21 | 1.61 / 1.64 | 5.39 / 5.65 | [Link](https://drive.google.com/drive/folders/14lO0xdOUR86AUbbqc7SSirbrwtNj5jSz?usp=drive_link) |

\* with Rawboost augmentation.


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
@inproceedings{huang2026generalizable,
  title={Generalizable speech deepfake detection via information bottleneck enhanced adversarial alignment},
  author={Huang, Pu and Wang, Shouguang and Yao, Siya and Zhou, Mengchu},
  booktitle={ICASSP 2026-2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={19087--19091},
  year={2026},
  organization={IEEE}
}
```
