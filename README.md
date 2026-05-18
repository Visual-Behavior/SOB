# Towards Sharper Object Boundaries in Self-Supervised Depth Estimation

## Installation

```bash
# Install the package
pip install -e .
```

Download weights and splits: [drive](https://drive.google.com/drive/folders/1fTJTT597vC2yvWaFJK1qAxTRhielPvlJ?usp=sharing)

### Evaluation

```bash
python3 ./scripts/eval.py
```
Kitti dataset mush be available at `data/Kitti`

### Citation
BMVC 2025 Oral:
```
@inproceedings{Cecille_2025_BMVC,
author    = {Aurélien Cecille and Stefan Duffner and Franck Davoine and Rémi Agier and Thibault Neveu},
title     = {Towards Sharper Object Boundaries in Self-Supervised Depth Estimation},
booktitle = {36th British Machine Vision Conference 2025, {BMVC} 2025, Sheffield, UK, November 24-27, 2025},
publisher = {BMVA},
year      = {2025},
url       = {https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_1077/paper.pdf}
}
