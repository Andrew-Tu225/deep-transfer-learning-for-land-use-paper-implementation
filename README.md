# Deep Transfer Learning for Land Use and Land Cover Classification

A PyTorch reproduction of the paper:

> Naushad, R.; Kaur, T.; Ghaderpour, E. **Deep Transfer Learning for Land Use and Land Cover Classification: A Comparative Study.** *Sensors* 2021, 21, 8083. [doi:10.3390/s21238083](https://doi.org/10.3390/s21238083)

The paper PDF is included in [paper/sensors-21-08083.pdf](paper/sensors-21-08083.pdf).

## What This Project Does

Satellite image patches from the **EuroSAT RGB dataset** (27,000 Sentinel-2 images, 64×64 px, 10 classes) are classified into land-use/land-cover categories such as Forest, River, Residential, and Highway.

Rather than training a CNN from scratch, two ImageNet-pretrained networks — **VGG16** and **Wide ResNet-50** — are fine-tuned by freezing the convolutional backbone and training a new fully connected classifier head. Training is optimized with:

- Data augmentation (Gaussian blur, horizontal/vertical flips, rotation, resizing)
- Early stopping (patience 5, best weights kept)
- Gradient clipping (max norm 0.1)
- Adaptive learning rate (ReduceLROnPlateau, factor 0.1, patience 2)

## Target Results

The paper's benchmark on the EuroSAT RGB validation set (Table 2):

| Model | Augmentation | Accuracy |
|---|---|---|
| VGG16 | No | 98.14% |
| VGG16 | Yes | 98.55% |
| Wide ResNet-50 | No | 99.04% |
| Wide ResNet-50 | Yes | **99.17%** |

The goal of this reproduction is to land within a few tenths of a percent of these numbers and to reproduce the class-confusion patterns reported in the paper (e.g., River ↔ Highway, and confusion among the vegetation classes).

## Status

🚧 **Planning stage** — implementation has not started yet.


### Roadmap

- [ ] Task 1 — Dataset download and PyTorch loaders (75/25 split, batch size 64)
- [ ] Task 2 — Preprocessing + augmentation pipelines (with/without augmentation)
- [ ] Task 3 — Model builders: frozen VGG16 / Wide ResNet-50 with new FC head
- [ ] Task 4 — Training loop with all four enhancement techniques
- [ ] Task 5 — Evaluation: accuracy, loss/accuracy curves, confusion matrices
- [ ] Task 6 — Run the 4-experiment comparison matrix
- [ ] Task 7 — Analysis and comparison against the paper's results

## Planned Structure

```
paper/                  # The paper PDF
.claude/.session/       # Implementation plan and session docs
src/                    # Dataset, models, training, evaluation modules
notebooks/              # Exploration and result visualization
outputs/                # Checkpoints, curves, confusion matrices (gitignored)
```

## Requirements

- Python 3.10+
- PyTorch + torchvision
- A CUDA GPU is strongly recommended — the paper trained on a Tesla P100 at ~6 min/epoch (~2 h per full run). Kaggle or Colab work fine.

Setup and run instructions will be added as the implementation lands.

## References

- EuroSAT dataset: https://github.com/phelber/eurosat
- Original authors' notebooks: https://github.com/raoofnaushad/EuroSAT_LULC
- EuroSAT paper (Helber et al., 2019): [doi:10.1109/JSTARS.2019.2918242](https://doi.org/10.1109/JSTARS.2019.2918242)
