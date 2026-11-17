"""Run the clean / P1 / P2 protocols (and optionally corruption-aware recalibration) for one frozen backbone.

  clean : clean supports, clean queries
  P1    : clean supports, corrupted queries
  P2    : supports corrupted with the same operator and severity as the query, same corrupted queries
All three use thresholds calibrated once on clean normal images. With --recalibrate, every P1/P2 unit also
reports the operating point after recomputing the thresholds on the calibration images corrupted like the
query (all 50, "recal50", or the first 10, "recal10"), plus the score-shift measurements of the paper.

Example:
  python run_protocols.py --data_path ./data/mvtec_anomaly_detection --backbone dinov2
  python run_protocols.py --data_path ./data/mvtec_anomaly_detection --backbone dinov3 --shots 8 --recalibrate
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from utils import runtime  # noqa: F401  (must come before torch)
import numpy as np

from corruptions import operators
from corruptions.pipeline import clean_input, corrupted_input
from datasets.mvtec import load_masks, records, sample_split, train_rng_id, verify_files
from models.backbones import build
from models.patch_knn import PatchKNN
from utils import protocol as P

CALIBRATION_COUNT, SPLIT_OFFSET, SMALL_CALIBRATION = 50, 10000, 10


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_path", required=True, help="MVTec AD root (contains bottle/, screw/, ...)")
    ap.add_argument("--backbone", choices=["dinov2", "dinov3"], default="dinov2")
    ap.add_argument("--categories", nargs="+", default=["bottle", "screw", "cable"])
    ap.add_argument("--shots", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--corruptions", nargs="+", default=list(operators.CORRUPTIONS), choices=list(operators.CORRUPTIONS))
    ap.add_argument("--severities", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--recalibrate", action="store_true", help="also evaluate corruption-aware recalibration")
    ap.add_argument("--save_patches", action="store_true", help="also save raw patch distances (.npy)")
    ap.add_argument("--verify_data", action="store_true", help="check images against the paper's SHA-256 manifest")
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dinov2_repo", default="facebookresearch/dinov2", help="GitHub repo or local clone")
    ap.add_argument("--dinov2_checkpoint", default=None, help="optional local dinov2_vits14_pretrain.pth")
    return ap.parse_args()


def save(path: Path, payload: dict, patches: np.ndarray | None):
    if patches is not None:
        np.save(path.with_suffix(".npy"), patches.astype(np.float32), allow_pickle=False)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    os.replace(tmp, path.with_suffix(".json"))


def main():
    args = parse()
    device_info = runtime.configure(args.device)
    kwargs = {"repo": args.dinov2_repo, "checkpoint": args.dinov2_checkpoint} if args.backbone == "dinov2" else {}
    encoder = build(args.backbone, args.device, **kwargs)
    size, max_shots = encoder.input_size, max(8, max(args.shots))
    out = Path(args.out) / args.backbone
    (out / "units").mkdir(parents=True, exist_ok=True)
    (out / f"run_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json").write_text(json.dumps(
        {"args": vars(args), "device": device_info, "encoder": encoder.identity,
         "corruption_version": operators.VERSION, "corruption_config_sha256": operators.CONFIG_SHA256}, indent=1))
    t0 = time.perf_counter()

    def log(msg):
        print(f"{datetime.now():%H:%M:%S} {msg}", flush=True)

    for category in args.categories:
        train = records(args.data_path, category, "train")
        test = records(args.data_path, category, "test")
        if args.verify_data:
            log(f"{category}: {verify_files(Path(args.data_path), train + test)} files match the paper's manifest")
        masks = load_masks(test)
        labels = np.array([r.label for r in test], bool)
        split = {(s, k): sample_split(train, s, k, CALIBRATION_COUNT, max_shots, SPLIT_OFFSET)
                 for s in args.seeds for k in args.shots}
        support_pool = sorted({int(i) for (s, k), (sup, _) in split.items() for i in sup})
        calib_pool = sorted({int(i) for (_, cal) in split.values() for i in cal})
        needed = sorted(set(support_pool) | set(calib_pool))

        log(f"{category}: clean features for {len(needed)} train and {len(test)} test images")
        clean_train = {i: encoder.extract(clean_input(train[i], size)) for i in needed}
        clean_test = np.stack([encoder.extract(clean_input(r, size)) for r in test])

        # ---- clean protocol and clean-calibrated thresholds (one set per seed and shot count)
        state = {}
        for (s, k), (sup, cal) in split.items():
            detector = PatchKNN(np.stack([clean_train[int(i)] for i in sup]).reshape(-1, 384), args.device)
            cal_patches = np.stack([detector.patch_scores(clean_train[int(i)]) for i in cal])
            thr = P.thresholds_from(cal_patches)
            patches = np.stack([detector.patch_scores(f) for f in clean_test])
            metric, rows = P.evaluate(patches, masks, test, thr)
            cal_scores = P.image_scores(cal_patches)
            q_scores = P.image_scores(patches)
            state[(s, k)] = {"detector": detector, "thr": thr, "cal_scores": cal_scores, "q_scores": q_scores}
            save(out / "units" / f"{category}__clean0__k{k}__s{s}__clean",
                 {"category": category, "shots": k, "seed": s, "protocol": "clean", "corruption": "clean",
                  "severity": 0, "thresholds": thr, "support_ids": [train[int(i)].relative for i in sup],
                  "calibration_ids": [train[int(i)].relative for i in cal], "metrics": metric, "per_image": rows,
                  "encoder": encoder.identity}, patches if args.save_patches else None)
        log(f"{category}: clean done for {len(split)} seed/shot settings")

        # ---- corrupted protocols
        for name in args.corruptions:
            for severity in args.severities:
                tc = time.perf_counter()
                query = np.stack([encoder.extract(corrupted_input(r, r.image_id, name, severity, size)) for r in test])
                pool = needed if args.recalibrate else support_pool
                corrupted_train = {i: encoder.extract(corrupted_input(train[i], train_rng_id(train[i]), name, severity, size))
                                   for i in pool}
                for (s, k), (sup, cal) in split.items():
                    st = state[(s, k)]
                    banks = {"P1": st["detector"],
                             "P2": PatchKNN(np.stack([corrupted_train[int(i)] for i in sup]).reshape(-1, 384), args.device)}
                    mechanism = None
                    if args.recalibrate:
                        boot = P.bootstrap_threshold(st["cal_scores"], s)
                        cal_feats = [(clean_train[int(i)], corrupted_train[int(i)]) for i in cal]
                        mechanism = {
                            "feature_shift_cal": float(np.mean([1.0 - np.sum(c * x, axis=1, dtype=np.float64)
                                                                for c, x in cal_feats])),
                            "median_cal_score_clean": float(np.median(st["cal_scores"])),
                            "iqr_cal_score_clean": float(np.subtract(*np.percentile(st["cal_scores"], [75, 25]))),
                            "median_query_score_normal_clean": float(np.median(st["q_scores"][~labels])),
                            "median_query_score_anomalous_clean": float(np.median(st["q_scores"][labels])),
                            "tI_boot_sd": boot["sd"], "tI_boot_lo": boot["lo"], "tI_boot_hi": boot["hi"]}
                    for proto, detector in banks.items():
                        patches = np.stack([detector.patch_scores(f) for f in query])
                        metric, rows = P.evaluate(patches, masks, test, st["thr"])
                        payload = {"category": category, "shots": k, "seed": s, "protocol": proto,
                                   "corruption": name, "severity": severity, "thresholds": st["thr"],
                                   "support_ids": [train[int(i)].relative for i in sup], "metrics": metric,
                                   "per_image": rows, "encoder": encoder.identity}
                        if args.recalibrate:
                            cal_patches = np.stack([detector.patch_scores(x) for _, x in cal_feats])
                            qs, qm = P.image_scores(patches), P.anomaly_maps(patches)
                            recal = {}
                            for variant, n in (("recal50", len(cal_patches)), ("recal10", SMALL_CALIBRATION)):
                                thr = P.thresholds_from(cal_patches[:n])
                                recal[variant] = {"thresholds": thr, **P.operating_point(qs, qm, labels, masks, thr)}
                            payload["recalibration"] = recal
                            payload["score_shift"] = {
                                "median_score_normal": float(np.median(qs[~labels])),
                                "median_score_anomalous": float(np.median(qs[labels])),
                                "median_cal_score_corrupted": float(np.median(P.image_scores(cal_patches)))}
                            payload["mechanism"] = mechanism
                            if proto == "P1":
                                payload["fpr_at_bootstrap_hi"] = float((qs[~labels] > mechanism["tI_boot_hi"]).mean())
                            del qm
                        save(out / "units" / f"{category}__{name}{severity}__k{k}__s{s}__{proto}", payload,
                             patches if args.save_patches else None)
                log(f"{category} {name}{severity}: {2 * len(split)} units in {time.perf_counter() - tc:.0f}s "
                    f"(elapsed {(time.perf_counter() - t0) / 60:.1f} min)")
    log(f"done in {(time.perf_counter() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
