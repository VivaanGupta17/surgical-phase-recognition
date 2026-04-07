# SurgPhase — OR Deployment Guide

This document describes the technical, regulatory, and operational considerations for deploying SurgPhase as an intraoperative surgical AI tool in a clinical setting.

---

## Table of Contents

1. [Edge Hardware Deployment](#1-edge-hardware-deployment)
2. [Latency Requirements for Intraoperative Use](#2-latency-requirements-for-intraoperative-use)
3. [FDA Software as a Medical Device (SaMD) Considerations](#3-fda-samd-considerations)
4. [Model Versioning and PCCP Compliance](#4-model-versioning-and-pccp-compliance)
5. [Network and Integration Architecture](#5-network-and-integration-architecture)
6. [Failure Modes and Safety Design](#6-failure-modes-and-safety-design)
7. [Post-Market Surveillance](#7-post-market-surveillance)

---

## 1. Edge Hardware Deployment

### 1.1 Recommended Hardware

Surgical AI inference must run on hardware that can be physically co-located with the OR without introducing electromagnetic interference or infection risk. Two NVIDIA platforms are primary targets:

#### NVIDIA IGX Orin (Primary recommendation)

| Specification | Value |
|---------------|-------|
| SoC | NVIDIA Orin (12-core Arm Cortex-A78AE) |
| GPU | NVIDIA Ampere 2048 CUDA cores |
| Tensor Cores | 64 (4th generation) |
| INT8 TOPS | 200 |
| RAM | 32 GB LPDDR5 |
| Form factor | Enterprise AI workstation for industrial edge |
| Certification | IEC 62368-1 safety |

The IGX platform is specifically designed for deployment in safety-critical environments including medical devices. It supports [Functional Safety](https://developer.nvidia.com/embedded/jetson-agx-orin) (ISO 26262 and IEC 61508 partial compliance) and provides hardware-level watchdog timers.

#### NVIDIA Jetson AGX Orin (Developer / smaller footprint)

| Specification | Value |
|---------------|-------|
| GPU | 2048-core Ampere |
| INT8 TOPS | 275 |
| RAM | 32 GB LPDDR5 |
| Power envelope | 15W – 60W |
| Form factor | 100mm × 87mm module |

#### Minimum Hardware Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU VRAM | 4 GB | 8 GB |
| CPU RAM | 8 GB | 16 GB |
| Storage (NVMe) | 50 GB | 200 GB |
| Network | 1 Gbps | 10 Gbps |

### 1.2 TensorRT Optimisation for Edge

Export the trained model to TensorRT INT8 for edge inference:

```bash
# Step 1: Export to ONNX
python scripts/export_onnx.py \
    --checkpoint weights/best_model.pth \
    --output weights/surgphase.onnx \
    --opset 17 \
    --simplify

# Step 2: Build TensorRT engine (on target hardware)
python scripts/export_onnx.py \
    --checkpoint weights/best_model.pth \
    --output weights/surgphase.onnx \
    --tensorrt \
    --precision fp16 \
    --trt-output weights/surgphase_fp16.engine \
    --workspace-gb 4
```

**Expected inference performance after TensorRT optimisation:**

| Precision | IGX Orin FPS | Jetson AGX Orin FPS | Latency (ms) |
|-----------|-------------|---------------------|-------------|
| FP32 | 22 | 18 | 55.6 |
| FP16 | 41 | 31 | 32.3 |
| INT8 | 68 | 53 | 18.9 |

### 1.3 Docker Deployment

Use the NVIDIA L4T (Linux for Tegra) container for consistent edge deployment:

```dockerfile
FROM nvcr.io/nvidia/l4t-pytorch:r35.4.1-pth2.1-py3

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY configs/ ./configs/
COPY weights/ ./weights/
COPY scripts/ ./scripts/

EXPOSE 8080
CMD ["python", "scripts/demo_realtime.py", \
     "--source", "rtsp://OR_CAMERA:8554/endoscope", \
     "--onnx", "weights/surgphase_fp16.engine", \
     "--no-display"]
```

### 1.4 RTSP Stream Integration

Most surgical video towers (Karl Storz IMAGE1 S, Stryker 1588, Olympus OTV-S400) support RTSP output. Configure the endoscope tower to stream over the OR's isolated network:

```yaml
# Example stream configuration
stream:
  protocol: rtsp
  url: "rtsp://10.0.0.50:8554/scope"
  codec: H.264
  resolution: 1920x1080
  fps: 25
  latency_ms: 80  # Hardware encoding + network buffer
```

The SurgPhase pipeline introduces an additional 35–55 ms of model inference latency, for a total end-to-end latency of ~115–135 ms from endoscope to phase prediction.

---

## 2. Latency Requirements for Intraoperative Use

### 2.1 Clinical Latency Thresholds

Latency requirements are determined by the intended clinical use:

| Use Case | Acceptable Latency | Rationale |
|----------|-------------------|-----------|
| Phase-triggered alerts (passive advisory) | < 2000 ms | Surgeon has several seconds before next action |
| Instrument tray pre-positioning | < 5000 ms | Scrub tech needs lead time |
| Intraoperative documentation | < 30 000 ms | Async recording acceptable |
| Robotic arm auto-clutch gating | < 100 ms | Must respond before robot motion completes |
| Active instrument guidance overlay | < 50 ms | Must stay synchronised with surgeon's view |

### 2.2 Latency Budget Analysis

End-to-end latency from endoscope photons to phase prediction display:

```
Endoscope CCD → digitisation       ~5 ms
Video tower H.264 encoding         ~15 ms
Network (Gigabit Ethernet)         ~2 ms
GPU decode (H.264)                 ~5 ms
Frame buffer                       ~5 ms
Preprocessing (resize, normalize)  ~2 ms
Spatial encoder (ResNet50)         ~12 ms   ← TensorRT FP16
Temporal model (MS-TCN)            ~8 ms    ← TensorRT FP16
Post-processing + smoothing        ~3 ms
Display rendering                  ~5 ms
                                  ────────
Total                              ~62 ms   ✓ (< 100 ms target)
```

### 2.3 Graceful Degradation

When processing falls behind the target FPS:

1. **Frame dropping**: SurgPhase drops the oldest frames in the temporal buffer, not the latest. This keeps predictions based on the most recent context.
2. **Feature caching**: When using pre-extracted features, inference skips the spatial encoder entirely (60% latency reduction).
3. **Model downgrade**: Fall back from `surgphase-full` (EfficientNet-B4) to `surgphase-lite` (EfficientNet-B0) if FPS drops below 15.

---

## 3. FDA Software as a Medical Device (SaMD) Considerations

> **Disclaimer**: This section provides general guidance based on publicly available FDA guidance documents. It does not constitute legal or regulatory advice. Consult a qualified regulatory affairs professional before submission.

### 3.1 Device Classification

SurgPhase falls under FDA's SaMD framework. Determine your risk level using the IMDRF risk categorisation:

| Dimension | SurgPhase Classification |
|-----------|-------------------------|
| State of healthcare situation | Serious condition |
| Significance of SaMD output | Inform clinical management |
| **IMDRF Category** | **Category III** |

Category III SaMD (serious condition + inform clinical management) requires:
- Pre-submission (Q-submission) meeting with FDA
- **De Novo** classification request or **510(k)** clearance
- Substantial clinical evidence of safety and effectiveness

### 3.2 Predetermined Change Control Plan (PCCP)

Under the [FDA PCCP Guidance (2024)](https://www.fda.gov/media/172228/download), you may pre-specify future model updates that do not require a new 510(k) submission. The PCCP must define:

**3.2.1 Description of Modifications**

Specify which changes are covered:
- Retraining on additional cholecystectomy videos (same procedure, same institution)
- Backbone architecture changes within specified accuracy bounds
- Temporal model hyperparameter adjustments

**3.2.2 Modification Protocol**

For each covered modification, specify:
- Performance metrics to evaluate (accuracy ≥ 88%, mJaccard ≥ 80%)
- Test dataset (held-out Cholec80 videos 41–80, minimum)
- Statistical acceptance criteria (two one-sided t-tests, α = 0.05)
- Human factors verification (no increase in use errors)

**3.2.3 Impact Assessment**

Document how modifications will be assessed for patient safety impact before deployment.

### 3.3 Software Documentation Requirements

Per [FDA guidance 21 CFR Part 820](https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfcfr/CFRSearch.cfm?CFRPart=820) and IEC 62304:

| Document | Contents |
|----------|----------|
| Software Requirements Specification | Functional, performance, safety requirements |
| Software Architecture Document | Module decomposition, data flows |
| Software Design Document | Class/function level specifications |
| Risk Management File | ISO 14971 FMEA, hazard analysis |
| Verification & Validation Plan | Unit tests, integration tests, clinical validation |
| Algorithm Change Protocol | PCCP or equivalent |

### 3.4 Clinical Validation Requirements

A 510(k) for a surgical phase recognition aid will likely require:

1. **Analytical validation**: Performance on labeled video dataset (Cholec80 + institution-collected data)
2. **Clinical validation**: Prospective reader study comparing surgeon performance with vs. without the SaMD
3. **Cybersecurity documentation**: Threat model, SBOM, vulnerability management plan
4. **Human factors**: Use error analysis, summative usability study (IEC 62366)

---

## 4. Model Versioning and PCCP Compliance

### 4.1 Semantic Versioning for Medical AI

Use a three-tier versioning scheme that maps to regulatory impact:

```
MAJOR.MINOR.PATCH-build

MAJOR: Architecture changes requiring new 510(k)
MINOR: Retraining on new data — covered by PCCP
PATCH: Threshold/post-processing changes only
```

Example: `2.3.1-2024Q4` indicates second generation architecture, third dataset revision, first hotfix, Q4 2024 deployment.

### 4.2 Model Registry

Maintain a model registry with the following metadata for each deployed version:

```yaml
# model_registry/v2.3.1-2024Q4.yaml
model_id: "surgphase-v2.3.1-2024Q4"
created: "2024-11-15"
architecture:
  backbone: "resnet50"
  temporal: "mstcn_4stage"
  feature_dim: 2048
  num_classes: 7
training:
  dataset: "cholec80 + 50 in-house videos"
  num_training_frames: 1_243_800
  training_date: "2024-10-01"
validation:
  cholec80_accuracy: 0.914
  cholec80_jaccard: 0.852
  cholec80_macro_f1: 0.927
  validation_date: "2024-11-10"
regulatory:
  submission_type: "PCCP_update"
  pccp_version: "1.2"
  fda_decision_number: "K241234"
  cleared_date: "2024-11-01"
deployment:
  onnx_hash_sha256: "a3f1b2c4..."
  trt_engine_hash_sha256: "d4e5f6a7..."
checksum_algorithm: "SHA-256"
```

### 4.3 Audit Trail

All model predictions must be logged for post-market surveillance:

```python
# Every prediction is logged with:
{
    "timestamp_utc": "2024-11-15T14:32:11.342Z",
    "case_id": "anon_case_001",  # Anonymised
    "model_version": "2.3.1-2024Q4",
    "frame_idx": 8450,
    "predicted_phase": 2,
    "phase_probability": 0.934,
    "processing_latency_ms": 38.2,
    "hardware_id": "IGX_UNIT_003",
    "device_sw_version": "1.4.2"
}
```

Audit logs are written to an append-only local store and synced to a central clinical data repository via encrypted transport (TLS 1.3) with HIPAA-compliant storage.

### 4.4 Rollback Procedure

Maintain the two previous stable versions on the edge device. Rollback procedure:

```bash
# Verify current version
surgphase-ctl status

# Rollback to previous version
surgphase-ctl rollback --version 2.2.0-2024Q3

# Confirm rollback
surgphase-ctl verify-model
```

Rollback must complete within 5 minutes without rebooting OR systems.

---

## 5. Network and Integration Architecture

### 5.1 OR Network Isolation

Deploy SurgPhase on an isolated OR network segment (VLAN) with:
- No direct internet access
- Firewall rules allowing only approved clinical data flows
- IDS/IPS monitoring
- Encrypted connections only (TLS 1.3, no TLS 1.0/1.1)

### 5.2 Integration with OR Information Systems

SurgPhase can integrate with:
- **RTSP**: Endoscope video input (primary)
- **SDI/HDMI capture card**: Alternative for non-networked towers
- **HL7 FHIR**: Structured procedure documentation output
- **OR integration platforms**: Olympus Visera, Stryker SYSTEM1, Karl Storz OR1

---

## 6. Failure Modes and Safety Design

### 6.1 FMEA Summary

| Failure Mode | Effect | Probability | Severity | Mitigation |
|-------------|--------|-------------|----------|-----------|
| Frame drop > 50% | Stale prediction | Medium | Low | Show staleness indicator |
| GPU OOM | No prediction | Low | Medium | Pre-allocate memory, watchdog restart |
| Wrong phase prediction | Incorrect alert | Medium | High | CRF smoothing, confidence threshold |
| Network video loss | No input | Low | High | Freeze last prediction, show warning |
| Model file corruption | Startup failure | Very Low | Critical | SHA-256 checksum at startup |

### 6.2 Safety Design Principles

1. **Advisory only**: SurgPhase is a decision-support tool. All clinical decisions remain with the surgeon. The UI clearly labels predictions as "AI Suggestion."
2. **Confidence threshold**: Predictions with max softmax probability < 0.6 display as "Uncertain" rather than showing a phase label.
3. **Watchdog timer**: If inference latency exceeds 500 ms for 3 consecutive frames, the system alerts the operator and logs the event.
4. **Fail-safe**: On any unrecoverable error, SurgPhase disables its display output. The video tower continues to function normally.

---

## 7. Post-Market Surveillance

### 7.1 Performance Monitoring

Collect anonymised prediction data post-deployment:
- Compare predicted phase sequence against manual annotations (spot audits, 5% of cases)
- Track phase recognition accuracy over time — alert if accuracy drops > 5% from baseline
- Monitor latency distribution — alert if P95 latency exceeds 150 ms

### 7.2 Drift Detection

Surgical video appearance can change due to:
- Endoscope upgrades (new sensor characteristics)
- New OR lighting systems
- Changes in surgical technique

Implement distribution shift detection via:
- Monitor input feature statistics (mean, variance) per batch
- Alert if Mahalanobis distance of feature distribution exceeds trained baseline by > 3σ
- Trigger re-calibration or model update if drift persists > 30 days

### 7.3 Adverse Event Reporting

Under 21 CFR Part 803, report:
- Any case where SurgPhase displayed an incorrect phase during a critical surgical step
- Any device malfunction that could have contributed to patient harm
- All complaints logged with: case date, model version, error description, clinical context

---

*Document version: 1.0 | Last updated: 2024 | SurgPhase Regulatory Affairs Team*
