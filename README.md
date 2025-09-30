# IB-CAAN: Information Bottleneck enhanced Confidence-Aware Adversarial Network

Official PyTorch implementation of [Generalizable Speech Deepfake Detection via Information Bottleneck Enhanced Adversarial Alignment](https://arxiv.org/abs/2509.23618).

## 🌿 Branch Overview

- **master**: For *ASVspoof 2019 LA*, *ASVspoof 2021 LA and DF* and *In-the-Wild* datasets.

- **asvspoof5**: For *ASVspoof 5* dataset 👉 [asvspoof5 branch](https://github.com/763021701/IB-CAAN/tree/asvspoof5).

## 🔥 Update:
✔️ [TODO.]
🔥 [September 16, 2025] The code is currently being organized and will be updated soon.



# Preparation

## Dependencies

Install requirements
```bash
pip install -r requirements.txt
```

Install fairseq
```bash
cd fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1
(This fairseq folder can also be downloaded from https://github.com/pytorch/fairseq/tree/a54021305d6b3c4c5959ac9395135f63202db8f1)
pip install --editable ./
pip install -r requirements.txt
```

## Datasets

Our experiments are performed on *ASVspoof 2019 LA*, *ASVspoof 2021 LA and DF*, *In-the-Wild*, and *ASVspoof 5*. These datasets are available at the following links:

*ASVspoof 2019 LA*: [https://zenodo.org/records/6906306](https://zenodo.org/records/6906306)

*ASVspoof 2021 LA*: [https://zenodo.org/records/4837263](https://zenodo.org/records/4837263)

*ASVspoof 2021 DF*: [https://zenodo.org/records/4835108](https://zenodo.org/records/4835108)

*In-the-Wild*: [https://deepfake-total.com/in_the_wild](https://deepfake-total.com/in_the_wild)

*ASVspoof 5*: [https://zenodo.org/records/14498691](https://zenodo.org/records/14498691)


# Reference Repo
Thanks for following open-source projects:                                                                  

1. Rawboost: https://github.com/TakHemlata/RawBoost-antispoofing Paper: [[Rawboost]](https://arxiv.org/abs/2111.04433)
2. Nes2Net: https://github.com/Liu-Tianchi/Nes2Net_ASVspoof_ITW Paper: [[Nes2Net]](https://arxiv.org/abs/2504.05657)
3. RawBMamba: https://github.com/cyjie429/RawBMamba Paper: [[RawBMamba]](https://arxiv.org/abs/2406.06086)
4. Asvspoof 5 Baselines: https://github.com/asvspoof-challenge/asvspoof5 Paper: [[Asvspoof5]](https://arxiv.org/abs/2408.08739)

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