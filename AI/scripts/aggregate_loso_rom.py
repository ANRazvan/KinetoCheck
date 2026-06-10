#!/usr/bin/env python3
"""
Aggregate LOSO and ROM summary JSONs into per-exercise CSVs and a compact LaTeX table.

Writes:
- AI/checkpoints/aggregated_per_exercise_loso.csv
- AI/checkpoints/aggregated_per_exercise_rom.csv
- AI/checkpoints/loso_per_ex_table.tex

Run: python AI/scripts/aggregate_loso_rom.py
"""
import json
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path("AI/checkpoints")
LOSO = ROOT / "uiprmd_phase_aware_loso" / "all_exercises_summary.json"
ROM = ROOT / "uiprmd_phase_aware_rom" / "all_exercises_summary.json"


def load(p: Path):
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text())


def aggregate(rows):
    by_ex = defaultdict(list)
    for r in rows:
        ex = r.get("exercise_id")
        by_ex[ex].append(r)
    per = {}
    for ex, recs in sorted(by_ex.items()):
        def stats(key):
            vals = [float(r.get(key, float("nan"))) for r in recs]
            vals = [v for v in vals if not math.isnan(v)]
            if not vals:
                return (float("nan"), float("nan"))
            return (statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0)

        per[ex] = {
            "n_folds": len(recs),
            "f1_mean": stats("best_test_f1"),
            "acc_mean": stats("best_test_accuracy"),
            "prec_mean": stats("best_test_precision"),
            "rec_mean": stats("best_test_recall"),
            "loss_mean": stats("best_test_loss"),
        }
    return per


def overall_from_per(per):
    keys = ["f1_mean", "acc_mean", "prec_mean", "rec_mean", "loss_mean"]
    vals = {k: [v[k][0] for v in per.values()] for k in keys}
    out = {}
    for k, arr in vals.items():
        arr2 = [a for a in arr if not math.isnan(a)]
        out[k] = (statistics.mean(arr2), statistics.stdev(arr2) if len(arr2) > 1 else 0.0)
    return out


def write_csv_loso(per, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["exercise_id", "n_folds", "f1_mean", "f1_std", "acc_mean", "acc_std", "prec_mean", "prec_std", "rec_mean", "rec_std", "loss_mean", "loss_std"])
        for ex, vals in sorted(per.items()):
            w.writerow([
                ex,
                vals["n_folds"],
                f"{vals['f1_mean'][0]:.6f}", f"{vals['f1_mean'][1]:.6f}",
                f"{vals['acc_mean'][0]:.6f}", f"{vals['acc_mean'][1]:.6f}",
                f"{vals['prec_mean'][0]:.6f}", f"{vals['prec_mean'][1]:.6f}",
                f"{vals['rec_mean'][0]:.6f}", f"{vals['rec_mean'][1]:.6f}",
                f"{vals['loss_mean'][0]:.6f}", f"{vals['loss_mean'][1]:.6f}",
            ])


def write_csv_rom(per, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["exercise_id", "f1_mean", "acc_mean", "prec_mean", "rec_mean", "loss_mean"])
        for ex, vals in sorted(per.items()):
            w.writerow([
                ex,
                f"{vals['f1_mean'][0]:.6f}",
                f"{vals['acc_mean'][0]:.6f}",
                f"{vals['prec_mean'][0]:.6f}",
                f"{vals['rec_mean'][0]:.6f}",
                f"{vals['loss_mean'][0]:.6f}",
            ])


def write_tex(per, path: Path):
    # create a compact LaTeX table where each row ends with \\\\ (LaTeX row end)
    lines = ["\\begin{tabular}{c c c c c c}", "\\toprule", "Ex & F1 (mean±std) & Acc (mean±std) & Prec & Rec & Loss \\\\", "\\midrule"]
    for ex, vals in sorted(per.items()):
        # build the f-string without embedding the LaTeX row-end directly into the f-string
        row = (
            f"{ex} & {vals['f1_mean'][0]:.3f} $\\pm$ {vals['f1_mean'][1]:.3f} & "
            f"{vals['acc_mean'][0]:.3f} $\\pm$ {vals['acc_mean'][1]:.3f} & "
            f"{vals['prec_mean'][0]:.3f} & {vals['rec_mean'][0]:.3f} & {vals['loss_mean'][0]:.3f} "
        ) + "\\\\"
        lines.append(row)
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main():
    loso_rows = load(LOSO)
    rom_rows = load(ROM)

    loso_per = aggregate(loso_rows)
    rom_per = aggregate(rom_rows)

    write_csv_loso(loso_per, ROOT / "aggregated_per_exercise_loso.csv")
    write_csv_rom(rom_per, ROOT / "aggregated_per_exercise_rom.csv")
    write_tex(loso_per, ROOT / "loso_per_ex_table.tex")

    loso_overall = overall_from_per(loso_per)
    rom_overall = overall_from_per(rom_per)

    print("OVERALL (per-exercise means aggregated):")
    print("Metric | LOSO mean ± std | ROM mean ± std | delta (LOSO - ROM)")
    metric_names = [("f1_mean", "F1"), ("acc_mean", "Accuracy"), ("prec_mean", "Precision"), ("rec_mean", "Recall"), ("loss_mean", "Loss")]
    for key, label in metric_names:
        lmean, lstd = loso_overall[key]
        rmean, rstd = rom_overall[key]
        delta = lmean - rmean
        print(f"{label}: {lmean:.4f} ± {lstd:.4f} | {rmean:.4f} ± {rstd:.4f} | {delta:+.4f}")

    print("\nWrote: {} , {} , {}".format(ROOT / "aggregated_per_exercise_loso.csv", ROOT / "aggregated_per_exercise_rom.csv", ROOT / "loso_per_ex_table.tex"))


if __name__ == '__main__':
    main()
