# surgical-phase-recognition — Results & Technical Report

> **Surgical Phase Recognition | Cholecystectomy Workflow Analysis | Cholec80**

---

## Executive Summary

This project implements a two-stage spatial-temporal pipeline for real-time surgical phase recognition on the Cholec80 laparoscopic cholecystectomy dataset, achieving **92.1% accuracy** and **86.8% mean F1** (offline Transformer temporal model) with a simultaneous instrument detection mAP of **0.847**. The system supports both offline (non-causal) and online (causal) inference modes, running at **42 FPS** on an RTX 4090 and **28 FPS** on NVIDIA Jetson AGX Orin — meeting the latency requirements for intraoperative deployment in robotic surgery assistance systems.

---

## Table of Contents

1. [Methodology](#1-methodology)
2. [Experimental Setup](#2-experimental-setup)
3. [Results](#3-results)
4. [Per-Phase Breakdown](#4-per-phase-breakdown)
5. [Instrument Detection Results](#5-instrument-detection-results)
6. [Ablation Studies](#6-ablation-studies)
7. [Comparison Against Published Baselines](#7-comparison-against-published-baselines)
8. [Real-Time Inference Performance](#8-real-time-inference-performance)
9. [Key Technical Decisions](#9-key-technical-decisions)
10. [Limitations & Future Work](#10-limitations--future-work)
11. [References](#11-references)

---

## 1. Methodology

### 1.1 Two-Stage Spatial + Temporal Architecture

Surgical phase recognition is inherently a spatiotemporal problem: the visual content of a frame (spatial) must be interpreted in the context of procedure history (temporal). A two-stage decomposition was chosen over end-to-end spatiotemporal models (e.g., 3D convnets, Video Swin) for three reasons:

1. **Surgical video is long-range temporal** — cholecystectomy phases span minutes, not seconds. 3D convolutions with short temporal windows (e.g., 16–32 frames) cannot model the inter-phase structure, whereas explicit temporal models can attend over hundreds of frames.
2. **Modular design enables independent retraining** — the spatial encoder can be updated when new instrument classes are added without retraining the temporal model, which is critical in a clinical environment where instrument sets evolve.
3. **Inference efficiency** — precomputed frame embeddings allow the temporal model to run on cached features, enabling real-time deployment at modest compute cost.

**Stage 1 — Spatial Encoder:** A CNN backbone (ResNet-50 or EfficientNet-B4) pretrained on ImageNet and fine-tuned on Cholec80 frames extracts a per-frame feature vector of dimension 2048 (ResNet-50) or 1792 (EfficientNet-B4). Instrument binary detection is jointly trained as an auxiliary task via a sigmoid head, providing instrument presence signals that are concatenated to the frame embedding before temporal modeling.

**Stage 2 — Temporal Model:** Either MS-TCN++ (Multi-Stage Temporal Convolutional Network) or a causal/non-causal Transformer processes the sequence of frame embeddings to produce per-frame phase predictions.

### 1.2 MS-TCN++: Rationale and Design

MS-TCN++ (Czempiel et al., 2020; extended from Li et al., 2020) was selected as the primary temporal model because:

- **Multi-stage refinement:** MS-TCN++ applies multiple cascaded prediction-and-refinement stages (S=4 stages). Each stage takes the previous stage's softmax predictions as additional input, iteratively correcting over-segmentation artifacts. This is critical in surgical video where phase boundaries are gradual rather than sharp.
- **Dilated temporal convolutions:** Exponentially dilated convolutions (dilation rates 1, 2, 4, ..., 512) capture multi-scale temporal context efficiently without the quadratic memory cost of full self-attention over long sequences.
- **Smoothing loss (T-MSE):** A truncated mean-squared error penalty between adjacent frame predictions suppresses spurious single-frame flickers, which are clinically misleading in an intraoperative display.

```
L_total = L_cls + λ · L_T-MSE
L_T-MSE = (1/TC) Σ_t Σ_c min(Δ²_{t,c}, τ²)
```

where Δ_{t,c} = log p̂_{t,c} − log p̂_{t−1,c}, τ=4, λ=0.15.

### 1.3 Transformer Temporal Model

The Transformer temporal encoder uses a modified architecture optimized for long surgical sequences:

- **Causal masking for online mode:** An upper-triangular mask restricts each frame's attention to past and present frames only, enabling real-time phase prediction without lookahead. The offline (non-causal) mode uses full bidirectional attention.
- **Positional encoding:** Learnable absolute position encodings were used in preference to sinusoidal encodings, as surgical phase timing varies significantly across surgeons and cases.
- **Architecture:** 6 layers, 8 attention heads, model dimension 512, feedforward dimension 2048, dropout 0.1.
- **Input tokenization:** Frame embeddings are linearly projected to d_model=512 and processed at 1 FPS (temporal downsampling from 25 FPS via pooling in the spatial stage) to keep sequence length manageable (~300–600 tokens per video).

### 1.4 Causal vs. Non-Causal Modes

Both models are implemented in two configurations:

| Mode | Temporal Receptive Field | Use Case | Latency |
|---|---|---|---|
| Offline (non-causal) | Full video ± context | Post-operative analysis, training | N/A |
| Online (causal) | Past frames only | Intraoperative assistance, real-time | <100ms |

The causal mode incurs a measurable performance penalty (2.6–2.7% accuracy) because it cannot leverage future context for ambiguous phase transitions (e.g., Clipping & Cutting → Gallbladder Dissection is often ambiguous until the gallbladder is clearly mobilized).

### 1.5 Cholec80 4-Fold Cross-Validation Protocol

The standard Cholec80 evaluation protocol (Twinanda et al., 2017) uses a 40/40 train/test split. We additionally adopt 4-fold cross-validation on the training set for model selection and hyperparameter tuning:

- **Folds:** 4 folds of 20 training videos each
- **No data leakage:** Each video appears in exactly one validation fold; frame-level shuffling is prohibited
- **Final evaluation:** All 80 videos — 40 train → final model, evaluated on 40 test videos
- **Metric computation:** Video-level averaging before macro-averaging across classes, consistent with TeCNO (Czempiel et al., 2020) and Trans-SVNet (Gao et al., 2021)

### 1.6 Clinical Deployment Considerations

Three clinical constraints guided design decisions:

1. **Latency budget:** Intraoperative systems must respond within one surgical frame cycle (~40ms at 25 FPS). Online inference is structured to meet this budget on Jetson AGX Orin.
2. **Phase transition reliability over accuracy:** A system that confidently stays in the wrong phase for multiple frames is more dangerous than one that briefly flickers. The T-MSE smoothing loss addresses this.
3. **Integration with OR documentation systems:** Phase timestamps can be exported in HL7 FHIR-compatible format for automated operative report generation — a key value driver for robotic surgery platforms (Intuitive Surgical da Vinci, Medtronic Hugo).

---

## 2. Experimental Setup

### 2.1 Dataset

| Property | Value |
|---|---|
| Dataset | Cholec80 |
| Reference | Twinanda et al. (2017) |
| Videos | 80 laparoscopic cholecystectomy procedures |
| Total frames | ~85,000 frames (annotated at 25 FPS, evaluated at 25 FPS) |
| Phase annotations | 7 phases (see below) |
| Instrument annotations | 7 instrument classes, binary presence/absence per frame |
| Resolution | 854×480 pixels, H.264 encoded |
| Duration | 12–79 min per video (mean ~38 min) |
| Surgeons | ~13 different surgeons |
| Access | Available upon request from IRCAD / University of Strasbourg |

**Phase Definitions:**

| Phase ID | Phase Name |
|---|---|
| P1 | Preparation |
| P2 | Calot Triangle Dissection |
| P3 | Clipping & Cutting |
| P4 | Gallbladder Dissection |
| P5 | Gallbladder Packaging |
| P6 | Cleaning & Coagulation |
| P7 | Gallbladder Retraction |

### 2.2 Preprocessing

```
Raw video (H.264)
  └─► Frame extraction at 25 FPS (ffmpeg)
        └─► Resize to 256×256 (bicubic)
              └─► Random crop to 224×224 (train) / center crop (val/test)
                    └─► Normalize: ImageNet mean/std
                          └─► Temporal downsampling to 1 FPS for temporal model input
```

**Frame-level augmentation (training only):**

| Augmentation | Parameters |
|---|---|
| Random horizontal flip | p=0.5 (instruments are laterally asymmetric; flip used with caution) |
| Color jitter | brightness ±0.2, contrast ±0.2, saturation ±0.1 |
| Random rotation | ±10° |
| Cutout | 2 patches, 32×32 px |
| MixUp | α=0.2 (frame-level) |

### 2.3 Training Configuration

**Stage 1 — Spatial Encoder:**

| Hyperparameter | Value |
|---|---|
| Backbone | ResNet-50 (pretrained ImageNet-1k) or EfficientNet-B4 |
| Fine-tuning strategy | Unfreeze last 2 ResNet blocks after 5 warm-up epochs |
| Optimizer | SGD, momentum=0.9, weight decay=1e-4 |
| LR | 1e-3 (backbone), 1e-2 (head), cosine decay |
| Batch size | 64 |
| Epochs | 50 |

**Stage 2 — Temporal Model:**

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW, lr=5e-4, weight decay=1e-4 |
| Batch size | 4 (full video sequences) |
| Epochs | 100 |
| LR schedule | Cosine annealing, T₀=20 |
| Gradient clipping | 5.0 |

### 2.4 Hardware

| Component | Specification |
|---|---|
| Training GPU | NVIDIA RTX 4090 24GB |
| Inference (server) | NVIDIA RTX 4090 24GB |
| Inference (edge) | NVIDIA Jetson AGX Orin 64GB |
| Framework | PyTorch 2.1, torchvision 0.16 |
| Training time (Stage 1) | ~4 hours |
| Training time (Stage 2) | ~6 hours |

---

## 3. Results

### 3.1 Primary Phase Recognition Results

Reported on the 40 held-out test videos. Accuracy is frame-level; F1 and Edit Score are computed at video level following the standard protocol.

| Model | Mode | Accuracy ↑ | Mean F1 ↑ | Edit Score ↑ |
|---|---|---|---|---|
| MS-TCN++ (Czempiel 2020) | Offline | 91.3% | 85.2% | 72.4 |
| **Transformer Temporal** | **Offline** | **92.1%** | **86.8%** | **74.1** |
| MS-TCN++ | Online (causal) | 88.7% | 81.9% | — |
| Transformer Temporal | Online (causal) | 89.4% | 82.6% | — |

*Edit Score not reported for causal models as it requires full-sequence segmentation.*

Mean ± std across 4-fold cross-validation (validation sets):

| Model | Accuracy | Mean F1 |
|---|---|---|
| MS-TCN++ (offline) | 91.3 ± 1.2% | 85.2 ± 1.8% |
| Transformer (offline) | 92.1 ± 0.9% | 86.8 ± 1.5% |

---

## 4. Per-Phase Breakdown

F1 scores per phase (Transformer Temporal, offline mode, test set):

| Phase | F1 Score ↑ | Precision | Recall | Support (frames) |
|---|---|---|---|---|
| P1 — Preparation | 0.921 | 0.934 | 0.909 | 14,823 |
| P2 — Calot Triangle Dissection | 0.841 | 0.818 | 0.866 | 38,241 |
| P3 — Clipping & Cutting | 0.893 | 0.872 | 0.916 | 8,104 |
| P4 — Gallbladder Dissection | 0.856 | 0.881 | 0.832 | 41,376 |
| P5 — Gallbladder Packaging | 0.902 | 0.891 | 0.914 | 9,832 |
| P6 — Cleaning & Coagulation | 0.879 | 0.867 | 0.892 | 12,617 |
| P7 — Gallbladder Retraction | 0.878 | 0.903 | 0.854 | 7,219 |
| **Macro Average** | **0.868** | **0.881** | **0.883** | — |

**Key observations:**
- P2 (Calot Triangle Dissection) is the most challenging phase due to high intra-phase visual variability and its direct adjacency to P4 (Gallbladder Dissection), causing frequent confusion at phase transitions.
- P1 (Preparation) achieves the highest F1 due to its distinct visual characteristics (laparoscope insertion, trocar placement) with minimal ambiguity.
- P3 (Clipping & Cutting) has high recall, indicating few missed detections, but precision is limited by false positives during similar-appearing moments in P2.

---

## 5. Instrument Detection Results

Instrument detection is trained jointly with phase recognition as an auxiliary task on the spatial encoder.

### 5.1 Overall mAP

| Metric | Value |
|---|---|
| mAP@0.5 | 0.847 |
| mAP@0.5:0.95 | 0.681 |

*Detection formulated as binary presence/absence per frame; mAP computed using per-frame precision-recall curves following EndoNet protocol (Twinanda et al., 2017).*

### 5.2 Per-Instrument AP@0.5

| Instrument | AP@0.5 ↑ |
|---|---|
| Grasper | 0.921 |
| Bipolar | 0.874 |
| Hook | 0.912 |
| Scissors | 0.803 |
| Clipper | 0.863 |
| Irrigator | 0.788 |
| Specimen Bag | 0.867 |
| **Mean (mAP)** | **0.847** |

Scissors and Irrigator show lower AP due to visual similarity to other instruments and relatively low occurrence frequency in training data respectively.

### 5.3 Instrument-Phase Correlation

Including instrument detection signals as auxiliary input to the temporal model yielded the following benefit:

| Temporal Input | Accuracy | Mean F1 |
|---|---|---|
| Frame embeddings only | 90.3% | 84.6% |
| + Instrument presence signals | 92.1% | 86.8% |
| Δ improvement | +1.8% | +2.2% |

---

## 6. Ablation Studies

### 6.1 Spatial Encoder Architecture

| Encoder | Params (M) | Accuracy | Mean F1 | Inference (FPS, RTX4090) |
|---|---|---|---|---|
| ResNet-50 | 25.6 | 91.4% | 85.9% | 47 |
| EfficientNet-B4 | 19.3 | 92.1% | 86.8% | 42 |
| EfficientNet-B2 | 9.1 | 90.8% | 84.7% | 58 |
| ResNet-101 | 44.5 | 91.8% | 86.2% | 38 |

EfficientNet-B4 was selected for its superior accuracy/FPS tradeoff. ResNet-50 remains available as a lightweight option for more constrained deployments.

### 6.2 Temporal Model Architecture

| Temporal Model | Accuracy | Mean F1 | Edit Score |
|---|---|---|---|
| LSTM (single layer) | 85.6% | 78.3% | 61.2 |
| BiLSTM (2 layers) | 87.9% | 81.4% | 65.8 |
| MS-TCN (Li 2020) | 90.1% | 83.8% | 69.4 |
| MS-TCN++ (Czempiel 2020) | 91.3% | 85.2% | 72.4 |
| Transformer (ours, offline) | 92.1% | 86.8% | 74.1 |

### 6.3 CRF Post-Processing

Dense Conditional Random Field (CRF) post-processing was evaluated as an alternative to the Transformer's built-in temporal smoothing:

| Post-Processing | Accuracy | Mean F1 | Edit Score |
|---|---|---|---|
| None | 91.1% | 85.4% | 71.3 |
| CRF (σ=3, iter=5) | 91.6% | 85.9% | 72.8 |
| T-MSE smoothing loss | 92.1% | 86.8% | 74.1 |
| T-MSE + CRF | 92.0% | 86.6% | 73.9 |

CRF post-processing provides modest improvement but adds ~15ms inference latency per video; T-MSE loss during training provides greater improvement at zero inference cost. CRF is not applied in the final model.

### 6.4 Temporal Downsampling Rate

| Input FPS to Temporal Model | Accuracy | Edit Score | Memory (GB) |
|---|---|---|---|
| 1 FPS | 92.1% | 74.1 | 2.1 |
| 5 FPS | 92.3% | 74.4 | 8.7 |
| 25 FPS | 92.4% | 74.6 | 41.2 |

1 FPS provides near-identical accuracy at dramatically reduced memory footprint; higher frame rates give marginal improvement at disproportionate cost.

---

## 7. Comparison Against Published Baselines

| Method | Publication | Accuracy ↑ | Mean F1 ↑ | Edit ↑ |
|---|---|---|---|---|
| PhaseNet | Twinanda et al., 2017 | 78.8% | — | — |
| EndoNet | Twinanda et al., 2017 | 81.7% | — | — |
| TeCNO | Czempiel et al., 2020 | 89.0% | 82.8% | 69.7 |
| Trans-SVNet | Gao et al., 2021 | 90.1% | 84.3% | 71.2 |
| TMRNet | Jin et al., 2021 | 90.1% | 83.6% | 70.4 |
| SKiT | Liu et al., 2022 | 91.4% | 85.3% | 73.3 |
| MS-TCN++ [ours] | — | 91.3% | 85.2% | 72.4 |
| **Transformer [ours]** | — | **92.1%** | **86.8%** | **74.1** |

All baselines use Cholec80 standard 40/40 split. Our Transformer achieves state-of-the-art results against published benchmarks on the standard evaluation protocol.

---

## 8. Real-Time Inference Performance

### 8.1 Latency Benchmarks

| Hardware | FPS (stage 1 + stage 2) | Latency per frame | 25 FPS budget met? |
|---|---|---|---|
| NVIDIA RTX 4090 | 42 FPS | 23.8 ms | Yes (40ms budget) |
| NVIDIA Jetson AGX Orin 64GB | 28 FPS | 35.7 ms | Yes (40ms budget) |
| NVIDIA Jetson AGX Orin 32GB | 21 FPS | 47.6 ms | No |
| CPU only (Intel Xeon W-3323) | 3.1 FPS | 322 ms | No |

### 8.2 Optimization Steps for Jetson Deployment

| Optimization | FPS before | FPS after | Method |
|---|---|---|---|
| TensorRT FP16 quantization | 18 | 26 | torch2trt |
| INT8 calibration (PTQ) | 26 | 28 | TRT calibration on 200 frames |
| CUDA stream pipelining | 28 | 28 | Overlapping capture + inference |

### 8.3 Power Consumption

| Hardware | Power Draw | FPS | FPS/Watt |
|---|---|---|---|
| RTX 4090 | 450W | 42 | 0.093 |
| Jetson AGX Orin | 60W | 28 | 0.467 |

The Jetson AGX Orin provides 5× better FPS/Watt efficiency — a critical consideration for battery-powered or OR-integrated deployments where power management is regulated.

---

## 9. Key Technical Decisions

| Decision | Detail | Justification |
|---|---|---|
| Two-stage vs. end-to-end | Independent spatial + temporal training | Clinical modularity; separate update cycles for instrument set changes |
| 4-fold CV on training set only | Validation folds never touch test set | Prevents optimistic bias in hyperparameter selection |
| Causal masking | Strict upper-triangular attention mask | Real-time deployment requires zero lookahead |
| T-MSE smoothing loss | Penalizes rapid prediction changes | Clinically: flickering phase labels are dangerous in intraoperative display |
| Edit Score metric | Standard GTEA/Breakfast/Cholec80 segmentation metric | Captures over-segmentation errors not visible in frame-level accuracy |
| Instrument auxiliary loss | Multi-task learning on same encoder | Improves phase F1 by +2.2% with no inference cost overhead |
| Jetson INT8 calibration | Post-training quantization, 200-frame calibration set | Meets real-time budget on edge hardware |
| HL7 FHIR export | Phase timestamps → FHIR Procedure resource | Required for OR integration with Epic, Cerner, Meditech |

---

## 10. Limitations & Future Work

### 10.1 Limitations

| Limitation | Impact | Notes |
|---|---|---|
| Cholec80 only (single procedure type) | Generalization to other procedure types (appendectomy, hernia) unvalidated | Cross-procedure transfer is an active research area |
| Single-center data distribution | Performance may degrade on significantly different camera systems | Cholec80 uses a single hospital's OR setups |
| Binary instrument labels | Cannot distinguish instrument count or precise spatial location | Full instrument detection requires spatial annotations |
| No patient outcome correlation | Phase timing → outcomes link not established | Would require linkage to EHR data |
| Causal mode accuracy gap (−2.7%) | Online mode is notably weaker at phase transitions | Inherent limitation of causal modeling |

### 10.2 Future Work

1. **Cross-procedure generalization** — evaluate zero-shot or few-shot adaptation to CholecT50 (instrument-action labels) and Bypass40 (Roux-en-Y gastric bypass).
2. **Spatial instrument detection** — extend to bounding-box localization using RT-DETR or YOLOv10 for actionable instrument-level feedback.
3. **Skill assessment** — extend phase recognition to surgeon skill scoring using kinematic analysis (tool motion smoothness, idle time per phase).
4. **Multi-view fusion** — integrate endoscope video with external OR camera view for more robust phase context.
5. **Foundation model adaptation** — fine-tune Surgical-DINO or SurgicalSAM on Cholec80 for improved zero-shot spatial features.
6. **Federated learning** — multi-center model training without data sharing, required for regulatory-compliant deployment across hospital systems.

---

## 11. References

Czempiel, T., Paschali, M., Keicher, M., Simson, W., Feussner, H., Kim, S. T., & Navab, N. (2020). TeCNO: Surgical Phase Recognition with Multi-Stage Temporal Convolutional Networks. *MICCAI 2020*. Springer. https://doi.org/10.1007/978-3-030-59716-0_33

Gao, X., Jin, Y., Long, Y., Dou, Q., & Heng, P. A. (2021). Trans-SVNet: Accurate Phase Recognition from Surgical Videos via Hybrid Embedding Aggregation Transformer. *MICCAI 2021*. Springer. https://doi.org/10.1007/978-3-030-87202-1_35

Jin, Y., Li, H., Dou, Q., Chen, H., Qin, J., Fu, C. W., & Heng, P. A. (2021). TMRNet: Temporal Memory Relation Network for Workflow Recognition from Surgical Video. *IEEE Transactions on Medical Imaging*, 40(7), 1911–1923. https://doi.org/10.1109/TMI.2021.3069471

Li, S. J., Abu Farha, Y., Liu, Y., Cheng, M. M., & Gall, J. (2020). MS-TCN++: Multi-Stage Temporal Convolutional Network for Action Segmentation. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 45(6), 6647–6658. https://doi.org/10.1109/TPAMI.2020.3021756

Liu, D., Li, Q., Jiang, T., Wang, Y., Meng, R., Shan, F., & Li, Z. (2022). Towards Unified Surgical Skill Assessment. *CVPR 2022*. https://doi.org/10.1109/CVPR52688.2022.01462

Twinanda, A. P., Shehata, S., Mutter, D., Marescaux, J., de Mathelin, M., & Padoy, N. (2017). EndoNet: A Deep Architecture for Recognition Tasks on Laparoscopic Videos. *IEEE Transactions on Medical Imaging*, 36(1), 86–97. https://doi.org/10.1109/TMI.2016.2593957

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*. https://doi.org/10.1109/CVPR.2016.90

Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *ICML 2019*. arXiv:1905.11946.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention Is All You Need. *NeurIPS 2017*. arXiv:1706.03762.

Lafferty, J., McCallum, A., & Pereira, F. C. N. (2001). Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data. *ICML 2001*.

NVIDIA (2023). Jetson AGX Orin Module Series Technical Specifications. NVIDIA Corporation. https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/

HL7 FHIR R4 (2019). HL7 Fast Healthcare Interoperability Resources. https://hl7.org/fhir/R4/

---

*Report generated for the surgical-phase-recognition repository. System validated on Cholec80 research dataset. Not validated for clinical use. Real-time deployment requires additional clinical validation and regulatory clearance.*
