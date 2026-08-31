# DoseRAD2026: Photon Dose Prediction on MRI

Submission code for the photon/MRI task of the
[DoseRAD2026 Grand Challenge](https://doserad2026.grand-challenge.org/):
predicting a 3D radiation dose distribution (Geant4 Monte Carlo ground
truth) directly from a patient MRI volume and one VMAT control point's beam
geometry, in place of running a full Monte Carlo simulation.

## Approach

A 3D U-Net (5-level encoder/decoder, 16-32-64-128-256 channels, two residual
units per block, ~4.8M parameters, built on [MONAI](https://monai.io/)'s
`UNet`) takes a two-channel input, an MRI volume and a ray-traced beam-path
mask, and predicts a single-channel dose volume. This is the same
architecture and beam encoder used for this team's photon/CT submission;
only the imaging preprocessing changes for MRI, since MR intensity has no
fixed cross-patient calibration the way CT's Hounsfield units do: each
volume is normalized against its own 99th-percentile intensity
(`NormalizeMR` in `src/data/transforms.py`), and the body mask comes from
an empirically chosen intensity threshold (`BodyMaskMR`).

Training uses a masked high-dose MAE term aligned with the challenge's own
scoring metric, plus body-masked and outside-body-masked L1 regularizers
(`src/training/losses.py`), optimized with AdamW and cosine annealing under
mixed precision (`src/training/trainer.py`).

## Layout

```
src/                Data pipeline, beam encoder, model, losses, training loop, evaluation metrics
scripts/             train.py, evaluate_cloud.py: training and evaluation entry points
configs/             Training config (beam type, modality, hyperparameters)
app.py, process.py   Grand Challenge /health + /invoke submission server
Dockerfile           Container build (root-level, for Grand Challenge's repo-linked build)
```

## Reproducing

```bash
pip install -r requirements.txt
```

**Train:**
```bash
python scripts/train.py --config configs/task2_photon_mr.yaml
```

**Evaluate against local held-out patients:**
```bash
python scripts/evaluate_cloud.py --config configs/task2_photon_mr.yaml --checkpoint checkpoints/task2_photon_mr/best.pt
```

Training data (75 patients, paired CT + MRI + beam JSON + Geant4 beam-level
dose) is released by the challenge organizers on
[Zenodo](https://doi.org/10.5281/zenodo.19347848) and is not included here.

**Build and run the submission container:**
```bash
docker build --platform=linux/amd64 -f Dockerfile -t doserad2026_task2_photon_mr .
```
The container implements the platform's required `/health` + `/invoke` HTTP
API and the documented 10-slot batched image/metadata I/O contract.

## Weights

Trained checkpoints are not tracked in this repository. The one actually
submitted and scored is uploaded to the Grand Challenge platform separately
from the container image, per the platform's own `model.tar.gz` mechanism.

## License

CC BY-NC 4.0, matching the DoseRAD2026 dataset license.
