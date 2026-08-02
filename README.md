# Deep Transfer Learning for Land Use and Land Cover Classification

A PyTorch reproduction of the paper:

> Naushad, R.; Kaur, T.; Ghaderpour, E. **Deep Transfer Learning for Land Use and Land Cover Classification: A Comparative Study.** *Sensors* 2021, 21, 8083. [doi:10.3390/s21238083](https://doi.org/10.3390/s21238083)

The paper PDF is included in [paper/sensors-21-08083.pdf](paper/sensors-21-08083.pdf). A full walkthrough of the reading process, implementation, and discrepancy analysis is in **[report.md](report.md)**.

## What This Project Does

Satellite image patches from the **EuroSAT RGB dataset** (27,000 Sentinel-2 images, 64×64 px, 10 classes) are classified into land-use/land-cover categories such as Forest, River, Residential, and Highway.

Rather than training a CNN from scratch, two ImageNet-pretrained networks — **VGG16** and **Wide ResNet-50** — are fine-tuned by freezing the convolutional backbone and training a new fully connected classifier head. Training uses early stopping, gradient clipping (max norm 0.1), adaptive learning rate (`ReduceLROnPlateau`), and an optional data-augmentation pipeline.

## Results

All four experiments from the paper’s Table 2 were run on Colab (Tesla T4, seed 42). The pipeline trained end-to-end, but validation accuracy landed several points below the published benchmarks — and augmentation hurt rather than helped:

| Model | Augmentation | Paper | This reproduction | Δ |
|---|---|---|---|---|
| VGG16 | No | 98.14% | 94.01% | −4.13 pp |
| VGG16 | Yes | 98.55% | 89.54% | −9.01 pp |
| Wide ResNet-50 | No | 99.04% | 92.80% | −6.24 pp |
| Wide ResNet-50 | Yes | **99.17%** | 90.61% | −8.56 pp |

Likely contributors include an incomplete augmentation set (no Gaussian blur), loss/head details taken from the authors’ notebooks, and a single train/val split instead of the paper’s five. See [report.md](report.md) for the full analysis and next steps.

## Requirements

- Python 3.10+
- PyTorch + torchvision
- A CUDA GPU is strongly recommended — the paper trained on a Tesla P100 at ~6 min/epoch (~2 h per full run). Kaggle or Colab work fine.

## References

- EuroSAT dataset: https://github.com/phelber/eurosat
- Original authors' notebooks: https://github.com/raoofnaushad/EuroSAT_LULC
- EuroSAT paper (Helber et al., 2019): [doi:10.1109/JSTARS.2019.2918242](https://doi.org/10.1109/JSTARS.2019.2918242)
