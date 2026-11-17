# Corruption-Induced Threshold Failure in Frozen-Feature Few-Shot Anomaly Detection

Official implementation of **Corruption-Induced Threshold Failure in Frozen-Feature Few-Shot Anomaly Detection** (under review).

![overview](assets/overview.png)

Frozen DINOv2/DINOv3 patch features with a cosine nearest-neighbour detector keep a high AUROC/AUPRO under image
corruption, yet their clean-calibrated thresholds can flag every normal image. This repository evaluates three
paired protocols on identical queries:

| Protocol | Supports | Queries | Thresholds |
|---|---|---|---|
| clean | clean | clean | calibrated on clean normal images |
| P1 | clean | corrupted | same clean thresholds |
| P2 | corrupted like the query | corrupted | same clean thresholds |

It also evaluates **corruption-aware recalibration** and the **score-shift measurements** behind the failure.

![pipeline](assets/pipeline.png)

## Install

```bash
conda create -n fsad_corruption python==3.11
conda activate fsad_corruption
bash install.sh
```

Tested with Python 3.11, PyTorch 2.11 (CUDA 12.8) on an NVIDIA RTX 3060 Laptop GPU (6 GB). All computation is
FP32 and deterministic (no TF32 or mixed precision).

## Data

Download [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) and extract it, e.g. to
`./data/mvtec_anomaly_detection` (folders `bottle/`, `screw/`, `cable/`, ...). No preprocessing is needed.
Pass `--verify_data` to check your copy against the SHA-256 manifest of the files used in the paper
(`configs/mvtec_ad_sha256.json`).

## Backbones

* **DINOv2 ViT-S/14** at 448 x 448, loaded with `torch.hub` from `facebookresearch/dinov2`
  (or a local clone via `--dinov2_repo`, and a local `dinov2_vits14_pretrain.pth` via `--dinov2_checkpoint`).
* **DINOv3 ViT-S/16** at 512 x 512, loaded with `timm` from the Hugging Face Hub
  (`timm/vit_small_patch16_dinov3.lvd1689m`; accept the DINOv3 license on the Hub first).

Both give a 32 x 32 grid of 384-d patch tokens. The checkpoint SHA-256 is printed if it differs from the paper's.

## Run

```bash
# full grid: 3 categories x 6 corruptions x 5 severities x k = 1, 2, 4, 8 x 5 seeds (clean, P1, P2)
python run_protocols.py --data_path ./data/mvtec_anomaly_detection --backbone dinov2

# eight-shot DINOv3 replication with corruption-aware recalibration and score-shift measurements
python run_protocols.py --data_path ./data/mvtec_anomaly_detection --backbone dinov3 --shots 8 --recalibrate

# paper-style tables (seed means over all conditions) -> results/<backbone>/summary.md and summary.csv
python summarize.py --results results/dinov2
```

Useful options: `--categories`, `--corruptions`, `--severities`, `--seeds`, `--shots`, `--save_patches`
(also store the raw 1,024 patch distances of every query), `--out`.

Each unit (category, condition, shot count, seed, protocol) is written to
`results/<backbone>/units/<category>__<condition>__k<k>__s<seed>__<protocol>.json` with its thresholds, support and
calibration images, metrics (image AUROC, pixel AUROC, AUPRO@0.3, image FPR/FNR, mean IoU) and per-image scores.
With `--recalibrate`, P1/P2 units also contain the operating point after recalibration on 50 and 10 corrupted
calibration images, and the median score shifts, feature shift and bootstrap threshold uncertainty.

### Check your installation

```bash
python run_protocols.py --data_path ./data/mvtec_anomaly_detection --backbone dinov2 --categories bottle --corruptions brightness --severities 4 --seeds 0 1 --shots 8 --recalibrate
python run_protocols.py --data_path ./data/mvtec_anomaly_detection --backbone dinov3 --categories screw --corruptions motion_blur --severities 5 --seeds 2 --shots 8 --recalibrate
python check_reproduction.py --results results
```

On our setup this code reproduces the paper's verified units exactly (difference 0, bit-identical patch scores).
Other GPUs or library versions may differ in the last digits.

## Repository structure

```
corruptions/   six native-resolution corruption operators (Gaussian noise, Gaussian blur, motion blur,
               brightness, contrast, JPEG) with per-image deterministic seeds; severities in configs/
datasets/      MVTec AD records, masks and seeded support/calibration splits
models/        frozen DINOv2/DINOv3 encoders and the cosine 1-NN patch detector (scores and anomaly maps)
utils/         metrics (exact AUROC, AUPRO@0.3), calibration, operating point, deterministic runtime
run_protocols.py      clean / P1 / P2 (+ recalibration) for one backbone
summarize.py          tables in the paper's aggregation
check_reproduction.py comparison with reference units from the paper
```

## Citation

The citation will be added after the review period.

## Acknowledge

We thank the authors of [DINOv2](https://github.com/facebookresearch/dinov2),
[DINOv3](https://github.com/facebookresearch/dinov3), [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)
and [ImageNet-C](https://github.com/hendrycks/robustness), whose corruptions our operators adapt.
