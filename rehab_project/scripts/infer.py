from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

# Ensure src/ is importable when running `python scripts/infer.py` from rehab_project/
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.factory import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for a single skeleton sequence.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(_PROJECT_ROOT / "configs" / "config.yaml"),
        help="Path to Hydra-composed config YAML.",
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input sequence (.npy, .npz, or .json) with shape [T, J, C].",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default="",
        help="Optional reference sequence path for direct joint deviation metrics.",
    )
    parser.add_argument("--device", type=str, default="", help="Override device: cpu or cuda.")
    return parser.parse_args()


def resolve_device(device_override: str, cfg: DictConfig) -> torch.device:
    if device_override:
        device_name = device_override
    else:
        device_name = str(cfg.project.get("device", "cuda"))

    if device_name.lower() == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_config(path: str) -> DictConfig:
    cfg = OmegaConf.load(path)
    if not isinstance(cfg, DictConfig):
        raise ValueError("Config must be a DictConfig-compatible YAML file.")
    return cfg


def load_sequence(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input sequence not found: {path}")

    if p.suffix.lower() == ".npy":
        seq = np.load(p)
    elif p.suffix.lower() == ".npz":
        archive = np.load(p)
        if "keypoints" in archive:
            seq = archive["keypoints"]
        else:
            # Use first item if standardized key is missing.
            first_key = list(archive.keys())[0]
            seq = archive[first_key]
    elif p.suffix.lower() == ".json":
        with p.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict) and "keypoints" in payload:
            seq = np.asarray(payload["keypoints"], dtype=np.float32)
        else:
            seq = np.asarray(payload, dtype=np.float32)
    else:
        raise ValueError("Unsupported input format. Use .npy, .npz, or .json.")

    seq = np.asarray(seq, dtype=np.float32)
    if seq.ndim != 3:
        raise ValueError(f"Expected sequence shape [T, J, C], got {seq.shape}.")
    return seq


def load_symbol(target: str) -> Any:
    if "." not in target:
        raise ValueError(f"Invalid target '{target}'. Expected dotted path.")

    module_path, symbol_name = target.rsplit(".", 1)
    module = importlib.import_module(module_path)

    try:
        return getattr(module, symbol_name)
    except AttributeError as exc:
        raise ValueError(f"Symbol '{symbol_name}' not found in '{module_path}'.") from exc


def build_preprocessor(cfg: DictConfig):
    if "preprocessing" not in cfg:
        raise ValueError("Missing 'preprocessing' section in config.")

    prep_cfg = cfg.preprocessing
    target = prep_cfg.get("target")
    if not target:
        raise ValueError("preprocessing.target is required to build the preprocessing pipeline.")

    prep_cls = load_symbol(str(target))
    prep_kwargs = OmegaConf.to_container(prep_cfg, resolve=True)
    if not isinstance(prep_kwargs, dict):
        raise ValueError("preprocessing config must resolve to a dictionary.")

    return prep_cls(prep_kwargs)


def preprocess_sequence(preprocessor, sequence: np.ndarray) -> np.ndarray:
    sample = {"keypoints": sequence}
    transformed = preprocessor(sample)["keypoints"]
    if transformed.ndim != 3:
        raise ValueError("Preprocessor output must have shape [T, J, C].")
    return transformed.astype(np.float32, copy=False)


def predict(model: torch.nn.Module, seq: np.ndarray, device: torch.device) -> tuple[int, np.ndarray]:
    x = torch.from_numpy(seq).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)

    pred_class = int(torch.argmax(probs, dim=-1).item())
    pred_probs = probs.squeeze(0).detach().cpu().numpy()
    return pred_class, pred_probs


def _align_temporal(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    target_t = min(a.shape[0], b.shape[0])

    def _resample(seq: np.ndarray, t: int) -> np.ndarray:
        if seq.shape[0] == t:
            return seq
        idx = np.linspace(0, seq.shape[0] - 1, t).astype(np.int64)
        return seq[idx]

    return _resample(a, target_t), _resample(b, target_t)


def compute_joint_deviation_metrics(
    processed_seq: np.ndarray,
    reference_seq: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute interpretable joint-deviation metrics.

    If reference_seq is provided, deviations are measured frame-wise against it.
    Otherwise, deviations are measured against the first frame baseline.
    """
    if reference_seq is not None:
        a, b = _align_temporal(processed_seq, reference_seq)
        if a.shape[1:] != b.shape[1:]:
            raise ValueError(
                "Processed and reference sequences must have matching [J, C] after preprocessing."
            )
        deltas = a - b
        mode = "reference"
    else:
        baseline = processed_seq[0:1]
        deltas = processed_seq - baseline
        mode = "first_frame"

    # Euclidean deviation per joint per frame -> [T, J]
    joint_frame_dev = np.linalg.norm(deltas, axis=-1)

    per_joint_mean = joint_frame_dev.mean(axis=0)
    per_joint_max = joint_frame_dev.max(axis=0)

    metrics = {
        "deviation_mode": mode,
        "num_frames": int(joint_frame_dev.shape[0]),
        "num_joints": int(joint_frame_dev.shape[1]),
        "global_mean_deviation": float(joint_frame_dev.mean()),
        "global_std_deviation": float(joint_frame_dev.std()),
        "global_max_deviation": float(joint_frame_dev.max()),
        "per_joint_mean_deviation": per_joint_mean.round(6).tolist(),
        "per_joint_max_deviation": per_joint_max.round(6).tolist(),
        "top3_joint_indices_by_mean_deviation": np.argsort(-per_joint_mean)[:3].astype(int).tolist(),
    }
    return metrics


def load_checkpoint_into_model(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: torch.device,
) -> dict[str, Any]:
    ckpt = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        raise KeyError("Checkpoint missing model weights ('model_state_dict' or 'state_dict').")

    model.load_state_dict(state_dict, strict=True)
    return ckpt


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)
    device = resolve_device(args.device, cfg)

    preprocessor = build_preprocessor(cfg)
    model = build_model(cfg).to(device)

    ckpt = load_checkpoint_into_model(model, args.checkpoint, device)

    raw_input_seq = load_sequence(args.input)
    proc_input_seq = preprocess_sequence(preprocessor, raw_input_seq)

    reference_seq = None
    if args.reference:
        raw_ref_seq = load_sequence(args.reference)
        reference_seq = preprocess_sequence(preprocessor, raw_ref_seq)

    pred_class, pred_probs = predict(model, proc_input_seq, device)
    deviation_metrics = compute_joint_deviation_metrics(proc_input_seq, reference_seq)

    output = {
        "device": str(device),
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": int(ckpt.get("epoch", -1)) if isinstance(ckpt, dict) else -1,
        "predicted_class": pred_class,
        "class_probabilities": pred_probs.round(6).tolist(),
        "joint_deviation_metrics": deviation_metrics,
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
