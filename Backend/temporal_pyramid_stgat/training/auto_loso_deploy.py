"""
Automatic LOSO training + deployment checkpoint selection.

What this script does:
1) Runs LOSO folds (one fold per subject) for a given exercise.
2) Saves per-fold summaries to CSV.
3) Saves aggregate mean/std metrics to CSV.
4) Selects best fold checkpoint (highest val_acc, then lowest val_loss).
5) Copies selected checkpoint to a deployment path for video assessment.

Usage:
    d:/Programming/KinetoCheck/.venv312/Scripts/python.exe -m temporal_pyramid_stgat.training.auto_loso_deploy --exercise 0 --epochs 30 --batch-size 16 --lr 0.001
"""

import os
import csv
import shutil
import argparse
import logging
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
import torch

from temporal_pyramid_stgat.config import PyramidSTGATConfig
from temporal_pyramid_stgat.preprocessing.uiprmd_loader import UIPRMDLoader
from temporal_pyramid_stgat.training.train_triplet import PyramidSTGATTripleLossTrainer


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def resolve_dataset_root(dataset_root: str) -> Path:
    """Resolve dataset root from common execution locations."""
    configured = Path(dataset_root)
    candidates = [
        configured,
        Path.cwd().parent / configured,
        Path(__file__).resolve().parents[3] / configured,
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    checked = "\n".join(str(c.resolve()) for c in candidates)
    raise FileNotFoundError(f"Could not find dataset root '{dataset_root}'. Checked:\n{checked}")


def discover_subject_ids(dataset_root: Path, exercise_id: int):
    """Read metadata once and discover unique subject IDs for LOSO."""
    loader = UIPRMDLoader(str(dataset_root))
    _, _, metadata = loader.load_all(exercise_id)
    subject_ids = sorted({int(m.get('subject_id', -1)) for m in metadata if int(m.get('subject_id', -1)) > 0})
    if not subject_ids:
        raise RuntimeError("No valid subject IDs discovered for LOSO.")
    return subject_ids


def write_csv(path: str, rows: list):
    if not rows:
        return
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Run LOSO automatically and export deployment checkpoint")
    parser.add_argument('--exercise', type=int, required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset-root', type=str, default='Datasets/UIPRMD')
    parser.add_argument('--model-save-dir', type=str, default='temporal_pyramid_stgat/weights')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset_root = resolve_dataset_root(args.dataset_root)
    subject_ids = discover_subject_ids(dataset_root, args.exercise)
    logger.info(f"Discovered LOSO subjects for exercise {args.exercise}: {subject_ids}")

    os.makedirs(args.model_save_dir, exist_ok=True)

    fold_rows = []

    for sid in subject_ids:
        logger.info(f"=== LOSO fold start: subject={sid} ===")

        config = PyramidSTGATConfig.for_uiprmd_single_exercise(args.exercise)
        config.dataset_root = args.dataset_root
        config.batch_size = args.batch_size
        config.epochs = args.epochs
        config.learning_rate = args.lr
        config.model_save_dir = args.model_save_dir
        config.checkpoint_name = f"pyramid_stgat_exercise_{args.exercise}_loso_s{sid}.pt"

        trainer = PyramidSTGATTripleLossTrainer(
            config,
            val_mode='loso',
            loso_subject=sid,
        )
        summary = trainer.train()

        fold_rows.append({
            'exercise': args.exercise,
            'subject': sid,
            'best_epoch_loss': summary['best_epoch_loss'],
            'best_val_loss': summary['best_val_loss'],
            'best_acc_at_loss_ckpt': summary['best_val_acc_for_loss_ckpt'],
            'best_epoch_acc': summary['best_epoch_acc'],
            'best_val_acc': summary['best_val_acc'],
            'best_loss_ckpt_path': summary['best_loss_ckpt_path'],
            'best_acc_ckpt_path': summary['best_acc_ckpt_path'],
            'metrics_csv_path': summary['metrics_csv_path'],
        })

        logger.info(f"=== LOSO fold done: subject={sid} | best_acc={summary['best_val_acc']:.2%}, best_loss={summary['best_val_loss']:.4f} ===")

    # Save per-fold metrics
    fold_csv = os.path.join(args.model_save_dir, f"loso_ex{args.exercise}_fold_metrics.csv")
    write_csv(fold_csv, fold_rows)
    logger.info(f"Saved fold metrics: {fold_csv}")

    # Aggregate mean/std
    losses = [float(r['best_val_loss']) for r in fold_rows]
    accs = [float(r['best_val_acc']) for r in fold_rows]
    agg_rows = [{
        'exercise': args.exercise,
        'num_folds': len(fold_rows),
        'mean_best_val_loss': mean(losses),
        'std_best_val_loss': pstdev(losses) if len(losses) > 1 else 0.0,
        'mean_best_val_acc': mean(accs),
        'std_best_val_acc': pstdev(accs) if len(accs) > 1 else 0.0,
    }]
    agg_csv = os.path.join(args.model_save_dir, f"loso_ex{args.exercise}_aggregate.csv")
    write_csv(agg_csv, agg_rows)
    logger.info(f"Saved aggregate metrics: {agg_csv}")

    # Pick deployment checkpoint: max val_acc, tie-breaker min val_loss
    best_row = sorted(
        fold_rows,
        key=lambda r: (-float(r['best_val_acc']), float(r['best_val_loss']))
    )[0]

    deployment_ckpt = os.path.join(
        args.model_save_dir,
        f"pyramid_stgat_exercise_{args.exercise}_deployment.pt"
    )

    src_ckpt = best_row['best_acc_ckpt_path']
    if not os.path.exists(src_ckpt):
        src_ckpt = best_row['best_loss_ckpt_path']

    shutil.copy2(src_ckpt, deployment_ckpt)
    logger.info(
        f"Deployment checkpoint selected from subject {best_row['subject']} | "
        f"acc={float(best_row['best_val_acc']):.2%}, loss={float(best_row['best_val_loss']):.4f}\n"
        f"Source: {src_ckpt}\n"
        f"Saved : {deployment_ckpt}"
    )

    # Save deployment selection metadata
    selection_csv = os.path.join(args.model_save_dir, f"loso_ex{args.exercise}_deployment_selection.csv")
    write_csv(selection_csv, [{
        'exercise': args.exercise,
        'selected_subject': best_row['subject'],
        'selected_val_acc': best_row['best_val_acc'],
        'selected_val_loss': best_row['best_val_loss'],
        'source_checkpoint': src_ckpt,
        'deployment_checkpoint': deployment_ckpt,
    }])
    logger.info(f"Saved deployment selection metadata: {selection_csv}")


if __name__ == '__main__':
    main()
