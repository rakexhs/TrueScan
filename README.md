# TrueScan

**Visual Synthetic Image Verification System**

TrueScan is an image-level deepfake verification pipeline focused on synthetic face detection under realistic compression and distribution shift.  
The repository is structured so another researcher can run the same workflow, regenerate artifacts, and audit each stage from preprocessing to final evaluation.

---

## Project Scope

TrueScan performs:

- Face-centric preprocessing with robust fallback logic
- Spatial and spatial-frequency model training
- In-distribution evaluation on FF++ c23
- Zero-shot external evaluation on Celeb-DF v2
- Robustness testing under common image degradations
- Post-hoc calibration and abstention analysis
- Grad-CAM visual explanation generation

The primary executable artifact is:

- `TrueScan.ipynb`

---

## Repository Layout

```text
TrueScan/
├── TrueScan.ipynb
└── outputs/
    ├── reports/
    ├── tables/
    ├── plots/
    └── gradcam/
```

Important outputs:

- `outputs/reports/environment_summary.txt` - runtime and library versions used
- `outputs/reports/final_results_summary.txt` - consolidated final metrics and findings
- `outputs/reports/inference_policy_config.json` - triage and publish-threshold policy
- `outputs/tables/final_metrics_table.csv` - headline quantitative results
- `outputs/tables/ffpp_test_metrics.csv` - in-distribution test metrics
- `outputs/tables/celebdf_external_metrics.csv` - external zero-shot metrics
- `outputs/tables/per_manipulation_metrics.csv` - per-manipulation analysis
- `outputs/tables/robustness_results.csv` - robustness under perturbations
- `outputs/tables/calibration_results.csv` - calibration before/after temperature scaling
- `outputs/tables/abstention_coverage_risk.csv` - selective prediction curve data

---

## Environment and Dependencies

Reference environment used in final execution:

- Python 3.12.13
- PyTorch 2.10.0 + CUDA 12.8
- Torchvision 0.25.0
- NumPy 2.0.2
- Pandas 2.2.2
- OpenCV 4.13.0
- TIMM 1.0.26
- scikit-learn 1.6.1

Core Python dependencies:

- `torch`, `torchvision`, `torchaudio`
- `numpy`, `pandas`, `scikit-learn`, `scipy`
- `opencv-python`, `Pillow`, `matplotlib`, `seaborn`
- `timm`
- `facenet-pytorch`
- `pytorch-grad-cam`
- `tqdm`, `pyyaml`

Create an environment (example with `venv`):

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install numpy pandas scikit-learn scipy opencv-python pillow matplotlib seaborn timm facenet-pytorch pytorch-grad-cam tqdm pyyaml
```

If you run on CPU-only, training will be slower and final numbers may vary slightly.

---

## Dataset Access and Preparation

This repository does not redistribute FF++ or Celeb-DF data.

Required datasets:

- **FaceForensics++ (c23)**:
  - Original (real)
  - Deepfakes
  - Face2Face
  - FaceSwap
  - NeuralTextures
- **Celeb-DF v2** for external zero-shot evaluation

Setup steps:

1. Download datasets from official sources according to their licenses/terms.
2. Place datasets in local storage paths used by the notebook configuration cells.
3. Run notebook dataset-check sections first to validate folder presence and counts.

The notebook includes data auditing utilities and writes:

- dataset status tables
- missingness/readability audit tables
- split summaries

---

## End-to-End Reproduction

1. Open `TrueScan.ipynb`.
2. Run all cells in order (top to bottom).
3. Do not skip configuration/check cells; they establish paths, seeds, and guards.
4. Ensure all major sections complete:
   - environment setup and imports
   - dataset verification and metadata checks
   - deterministic split construction
   - frame extraction and face-crop preprocessing
   - model training (baseline + main variants)
   - FF++ and Celeb-DF evaluation
   - calibration and abstention analysis
   - robustness sweeps
   - Grad-CAM generation
5. Confirm regenerated artifacts under `outputs/`.

Re-run guidance:

- Keep random seed fixed (`42`) for closest reproducibility.
- Re-run complete notebook for consistent artifacts.
- If hardware differs, compare trends and relative behavior, not only exact decimals.

---

## Method and Pipeline Summary

### Preprocessing

- Uniform frame sampling from source videos
- Face detection via MTCNN (`facenet-pytorch`) with fallback crop logic
- Standardized crop resizing and normalization
- Auxiliary frequency representation generation (DCT pathway input)

### Model Design

- Image-only baseline: EfficientNet-B0
- Main models: Xception + DCT branch, EfficientNet-B3 + DCT branch
- Fusion strategy combines spatial and frequency features for final classification

### Training Procedure

- Transfer learning and staged fine-tuning
- Data augmentation including compression/noise variations
- BCE/Focal-style objective settings from notebook config
- EMA, TTA, and snapshot ensembling in final evaluation path

### Evaluation Setup

- FF++ c23 in-distribution metrics
- Celeb-DF v2 external zero-shot metrics
- Per-manipulation analysis
- Robustness perturbation tests
- Temperature scaling calibration
- Abstention coverage-risk analysis
- Grad-CAM qualitative evidence

---

## Expected Outputs and Results

Representative headline values from the final run:

- FF++ snapshot ensemble AUROC: **0.9351**
- FF++ snapshot ensemble PR-AUC: **0.9829**
- FF++ primary model F1: **0.9159**
- Celeb-DF external AUROC: **0.8087**
- Celeb-DF external F1: **0.7550**
- Calibration temperature: **0.787**

Operational policy file:

- `outputs/reports/inference_policy_config.json`
  - triage: low/high thresholds for likely authentic vs likely synthetic
  - binary publish threshold: `0.22`

---

## Troubleshooting

- **Dataset not found**: verify notebook path config and dataset extraction.
- **Face detector unavailable**: ensure `facenet-pytorch` is installed; restart kernel.
- **CUDA issues**: check `torch.cuda.is_available()` and matching CUDA/PyTorch build.
- **OOM during training**: reduce batch size in config cells.
- **Metrics drift**: confirm seed, versions, and preprocessing path parity.

For auditability, treat `outputs/` as the source-of-truth artifact store for reproduced runs.

