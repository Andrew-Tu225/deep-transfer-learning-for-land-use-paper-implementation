# Reproduction Report: Deep Transfer Learning for Land Use and Land Cover Classification

**Paper:** Naushad, R.; Kaur, T.; Ghaderpour, E. *Deep Transfer Learning for Land Use and Land Cover Classification: A Comparative Study.* Sensors 2021, 21, 8083. ([doi:10.3390/s21238083](https://doi.org/10.3390/s21238083), [local PDF](paper/sensors-21-08083.pdf))

**Goal of this write-up:** Document how I read and understood the paper, how I reproduced the experiments, and where my results diverge from the published benchmarks.

---

## 1. Introduction

This project reproduces a transfer-learning study on the **EuroSAT RGB** dataset: 27,000 Sentinel-2 scene patches (64×64 px) labeled into 10 land-use / land-cover (LULC) classes. The paper fine-tunes ImageNet-pretrained **VGG16** and **Wide ResNet-50**, and reports a best validation accuracy of **99.17%** (Wide ResNet-50 with data augmentation).

I implemented the pipeline in PyTorch, ran all four experiments on Google Colab (Tesla T4), and compared against Table 2 of the paper. The pipeline trained successfully end-to-end, but validation accuracies landed roughly **4–9 percentage points below** the published numbers — and, unlike the paper, augmentation *hurt* rather than helped. The rest of this report walks through that process and the likely sources of the gap.

---

## 2. Reading the Paper: What Problem Are We Solving?

### 2.1 Why scene-level CNNs, not pixel/object methods?

Land-use and land-cover classification from satellite imagery is often approached with **pixel-based** or **object-based** algorithms. Those methods lean on local spectral or geometric detail. For *scene* imagery — small patches that represent a semantic category such as “Residential,” “Highway,” or “SeaLake” — the decision depends more on **high-level spatial context** than on per-pixel detail. Classical low-level pipelines struggle to capture that semantics.

That is why this paper (and this reproduction) focuses on **convolutional neural networks**. CNNs dominate image classification because they learn a hierarchy of features: early layers respond to edges, textures, and colors; deeper layers compose those into object- and scene-level patterns. For EuroSAT-style LULC, that hierarchy is a better match than hand-crafted pixel/object features alone.

### 2.2 Transfer learning setup

Training a deep CNN from scratch on 27,000 small patches is possible but wasteful when large natural-image corpora already exist. The paper uses **transfer learning**:

1. Start from a network pretrained on ImageNet.
2. **Freeze the convolutional backbone** so the pretrained feature extractors stay fixed.
3. **Replace the final fully connected (classifier) layers** with a new head tailored to the 10 EuroSAT classes.
4. Train only that new head on EuroSAT.

A useful clarification relative to a common shorthand: freezing the conv layers does *not* mean “we keep only high-level features and discard low-level ones.” The frozen stack preserves the **entire pretrained hierarchy** — low-level filters through high-level representations. What we discard and retrain is the **task-specific classifier**, so the network can map those ImageNet-derived features onto LULC labels.

### 2.3 Enhancement techniques and target results

Beyond the basic freeze-and-retrain recipe, the paper compares runs with and without several training enhancements:

- Data augmentation (Gaussian blur, horizontal/vertical flips, rotation, resizing)
- Gradient clipping (max total norm 0.1)
- Early stopping (patience 5, restore best validation weights)
- Adaptive learning rate (`ReduceLROnPlateau`, factor 0.1, patience 2)

Hyperparameters shared across experiments include: 75/25 train/val split, batch size 64, Adam with max LR \(1 \times 10^{-4}\), cross-entropy loss, max 25 epochs, and resize from 64×64 to 224×224 with ImageNet normalization.

**Paper Table 2 (validation accuracy):**

| Model | Augmentation | Accuracy | Epochs (paper) |
|---|---|---|---|
| VGG16 | No | 98.14% | 18 |
| VGG16 | Yes | 98.55% | 21 |
| Wide ResNet-50 | No | 99.04% | 14 |
| Wide ResNet-50 | Yes | **99.17%** | 23 |

The paper also evaluates over **five different random 75/25 splits** and reports confusion patterns (e.g., near-perfect Forest and Sea/Lake; confusion among vegetation classes; River ↔ Highway).

### 2.4 Ambiguities left by the paper text

While reading, several details needed for a byte-faithful reimplementation were underspecified:

- Exact fully connected head widths and dropout rate
- Precise augmentation hyperparameters (blur kernel, rotation degrees, whether “resizing” is a random crop/scale or just the fixed 64→224 resize)
- Exactly which VGG classifier layers are replaced vs. frozen
- Whether early stopping monitors validation loss or accuracy

I resolved these by consulting the authors’ reference notebooks: [raoofnaushad/EuroSAT_LULC](https://github.com/raoofnaushad/EuroSAT_LULC), and by encoding the clear hyperparameters in `src/config.py`.

---

## 3. Understanding → Design Decisions

With the paper’s intent clear, I structured the reproduction as small PyTorch modules driven by Colab notebooks (so training runs on a GPU without a local CUDA setup).

| Module | Role |
|---|---|
| `src/config.py` | Single source of truth for paper hyperparameters |
| `src/data.py` | EuroSAT loaders, 75/25 split, with/without-aug transforms |
| `src/models.py` | Frozen VGG16 / Wide ResNet-50 builders + new head |
| `src/training_loop.py` | Epoch loop, grad clip, scheduler, early stopping |
| `src/early_stopping.py` | Patience / best-weights tracking |
| `src/vgg_training.ipynb` | Colab entrypoint for both VGG experiments |
| `src/wide_resnet_training.ipynb` | Colab entrypoint for both Wide ResNet experiments |

**Classifier head** (taken from the authors’ notebooks, since the paper does not specify sizes):

```text
Linear(in → 256) → ReLU → Dropout(0.5) → Linear(256 → 10) → LogSoftmax
```

For VGG16, only `classifier[6]` is replaced; for Wide ResNet-50-2, `fc` is replaced. Everything else is frozen; only the new head trains.

**Loss pairing:** the head ends in `LogSoftmax` and is trained with `CrossEntropyLoss`. That combination is redundant (`CrossEntropyLoss` already applies log-softmax internally), but it matches the authors’ notebooks, so I kept it for fidelity rather than “fixing” it to `NLLLoss` or logits + CE without checking the effect.

**Compute:** paper used a Tesla P100 (~5.5–6 min/epoch). I used a Colab **Tesla T4**. That should affect wall-clock time more than final accuracy, but it is still a environmental difference worth noting.

---

## 4. Reproduction Process

### 4.1 Data and preprocessing

- Dataset: EuroSAT RGB (10 classes), loaded via `ImageFolder` / torchvision download in Colab.
- Split: random 75/25 (`VALID_SIZE=0.25`), seed **42**, batch size **64**.
- All images: resize to 224×224, convert to float tensor, ImageNet mean/std normalization.

### 4.2 Augmentation as implemented

When augmentation is enabled, the training pipeline adds:

- `RandomHorizontalFlip(p=0.5)`
- `RandomRotation(20)`
- `RandomVerticalFlip(p=0.5)`

Validation never uses augmentation. Relative to the paper’s listed set (Gaussian blur, H/V flip, rotation, resizing), **Gaussian blur is missing**, and “resizing” is only the fixed 64→224 step rather than an additional random resize/crop. That is one of the clearest implementation gaps.

### 4.3 Training recipe

For each of the four experiments:

1. Build the frozen pretrained model + new head.
2. Optimize with Adam (`lr=1e-4`) and `CrossEntropyLoss`.
3. Clip gradients to total norm 0.1 each step.
4. Step `ReduceLROnPlateau` on validation loss (factor 0.1, patience 2).
5. Early-stop on validation loss (patience 5); always restore best weights.
6. Cap training at 25 epochs.
7. Save `best.pt`, `metrics.json`, and plot loss / accuracy / LR curves.

### 4.4 Experiment matrix

| Run name | Model | Augmentation |
|---|---|---|
| `vgg16_noaug` | VGG16 | Off |
| `vgg16_aug` | VGG16 | On |
| `wide_resnet50_noaug` | Wide ResNet-50-2 | Off |
| `wide_resnet50_aug` | Wide ResNet-50-2 | On |

Artifacts from the Colab runs were written under Google Drive (`eurosat-lulc-outputs/…`). Curve plots remain embedded in the notebooks.

---

## 5. Results

All four runs completed on a Tesla T4 with seed 42. Early stopping **never fired**; every run used the full 25-epoch budget (the paper’s best checkpoints often came earlier).

| Experiment | Paper (Table 2) | This reproduction | Δ |
|---|---|---|---|
| VGG16, no aug | 98.14% | **94.01%** | −4.13 pp |
| VGG16, with aug | 98.55% | **89.54%** | −9.01 pp |
| Wide ResNet-50, no aug | 99.04% | **92.80%** | −6.24 pp |
| Wide ResNet-50, with aug | 99.17% | **90.61%** | −8.56 pp |

Wall-clock (approx.): VGG ~89–93 min/run; Wide ResNet ~97–100 min/run.

Two qualitative differences stand out immediately:

1. Absolute accuracy is several points below the paper across the board.
2. Augmentation **degrades** validation accuracy here, whereas the paper reports a small improvement for both architectures.

I have not yet produced confusion matrices, so I cannot confirm whether the paper’s class-confusion patterns (vegetation mix-ups, River ↔ Highway) also appear in this run.

---

## 6. Discrepancy Analysis

The gap is large enough that it is unlikely to be “random seed noise” alone. Below are the most plausible explanations, ordered by how strongly they seem supported by the current codebase and the paper text. These are hypotheses, not proven root causes.

### 6.1 Incomplete augmentation pipeline (high suspicion)

The paper lists Gaussian blur among the augmentations that help the with-aug setting reach 98.55% / 99.17%. My pipeline never applies blur, and the rotation/resize details may also differ from the authors’ notebooks. If the published with-aug gains depend on that full recipe, a partial pipeline would explain both (a) lower absolute numbers and (b) why augmentation hurt in my runs (weaker regularization / distribution shift without the balancing transforms).

### 6.2 Loss function pairing (medium–high suspicion)

`LogSoftmax` + `CrossEntropyLoss` applies log-softmax twice in effect. I kept this because the authors’ notebooks appear to do the same, but if the published numbers actually used logits + CE or `LogSoftmax` + `NLLLoss`, my training dynamics would be wrong. This is an easy A/B test for a follow-up.

### 6.3 VGG head attachment / freeze scope (medium suspicion for VGG)

I replace only `classifier[6]` and freeze the rest of VGG’s ImageNet FC stack (`25088 → 4096 → 4096`). The paper’s Figure 2 and the authors’ notebooks may instead replace the **entire** classifier. Leaving large frozen ImageNet FC layers in place changes what the new head sees and how many new parameters are truly task-adapted. Wide ResNet is less ambiguous (`fc` is a single layer), yet it still underperformed — so this cannot be the only issue.

### 6.4 Single split vs. five random splits (medium suspicion for variance)

The paper averages (or at least evaluates) over five random 75/25 splits. I used one fixed seed (42). That can move results by fractions of a point, but a **4–9 pp** gap is larger than typical split variance for this dataset size. Still, a multi-split rerun would make the comparison fairer.

### What matched vs. what drifted

**Matched (as far as the paper states clearly):**

- EuroSAT RGB, 10 classes, 75/25 split, batch 64
- Resize to 224 + ImageNet normalization
- Frozen backbone, trainable new head
- Adam \(1 \times 10^{-4}\), max 25 epochs
- Grad clip 0.1, early stopping patience 5, ReduceLROnPlateau factor 0.1 / patience 2
- Four-experiment matrix (2 models × aug on/off)

**Drifted or still ambiguous:**

- Augmentation set incomplete (no Gaussian blur; resize semantics unclear)
- Exact head / VGG classifier replacement strategy
- LogSoftmax + CrossEntropy redundancy
- One seed instead of five splits
- No confusion-matrix comparison yet
- Early stopping never triggered (all 25 epochs)

---

## 7. Lessons Learned and Next Steps

### Lessons

1. **Scene LULC needs semantic features.** Pixel/object pipelines are a poor fit when the label is a whole-patch category; CNN hierarchies address that directly.
2. **Transfer learning is freeze + swap.** Keep the pretrained conv feature hierarchy; replace the classifier so the decision layer fits EuroSAT.
3. **Paper text is not a full spec.** Head sizes, aug details, and freeze boundaries lived in the authors’ notebooks — and even after following those, a large gap remained.
4. **A runnable pipeline ≠ a matched benchmark.** All four experiments completed and produced smooth curves, but Table 2 was not reproduced within a few tenths of a percent.
5. **When augmentation hurts, treat the aug recipe as a first-class bug.** That qualitative inversion is as informative as the absolute accuracy gap.

### follow-ups

1. Align the augmentation pipeline with the authors’ notebooks (add Gaussian blur; verify rotation / resize).
2. A/B the loss: logits + `CrossEntropyLoss` vs. `LogSoftmax` + `NLLLoss` vs. current pairing.
3. For VGG, try replacing the entire `classifier` module rather than only `classifier[6]`.
4. Repeat each experiment across five random splits and report mean ± std.
5. Implement confusion matrices / per-class metrics and compare against the paper’s Figure 6 patterns.

A reproduction can be considered “closed” when validation accuracy is within a few tenths of Table 2 **and** the main confusion patterns match.

---

## 8. References

1. Naushad, R.; Kaur, T.; Ghaderpour, E. Deep Transfer Learning for Land Use and Land Cover Classification: A Comparative Study. *Sensors* 2021, 21, 8083. https://doi.org/10.3390/s21238083
2. Helber, P.; Bischke, B.; Dengel, A.; Borth, D. EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification. *IEEE J. Sel. Top. Appl. Earth Obs. Remote Sens.* 2019. https://doi.org/10.1109/JSTARS.2019.2918242
3. EuroSAT dataset: https://github.com/phelber/eurosat
4. Authors’ original notebooks: https://github.com/raoofnaushad/EuroSAT_LULC
5. This repository: `src/config.py`, `src/data.py`, `src/models.py`, `src/training_loop.py`, `src/vgg_training.ipynb`, `src/wide_resnet_training.ipynb`
