import argparse
import csv
import re
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize CLIP-ReID ablation train logs.")
    parser.add_argument("--results-root", default="results/ablation_loss_efficiency")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--target-ratio", type=float, default=0.95)
    return parser.parse_args()


def read_float(pattern: str, text: str):
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None


def read_text(pattern: str, text: str):
    match = re.search(pattern, text)
    return match.group(1).strip() if match else None


def summarize_log(path: Path, target_ratio: float) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore")
    speeds = [float(x) for x in re.findall(r"Speed:\s*([0-9.]+)\[samples/s\]", text)]
    time_per_batch = [float(x) for x in re.findall(r"Time per batch:\s*([0-9.]+)\[s\]", text)]

    evals = []
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        epoch_match = re.search(r"Cross-domain Results - Epoch:\s*(\d+)", line)
        if not epoch_match:
            continue
        window = "\n".join(lines[idx : idx + 5])
        map_match = re.search(r"mAP:\s*([0-9.]+)%", window)
        r1_match = re.search(r"Rank-1\s*:\s*([0-9.]+)%", window)
        if map_match:
            evals.append(
                {
                    "epoch": int(epoch_match.group(1)),
                    "mAP": float(map_match.group(1)),
                    "R1": float(r1_match.group(1)) if r1_match else None,
                }
            )

    best = max(evals, key=lambda row: row["mAP"]) if evals else {"epoch": None, "mAP": None, "R1": None}
    conv_epoch = None
    if evals and best["mAP"] is not None:
        target = best["mAP"] * float(target_ratio)
        for row in evals:
            if row["mAP"] >= target:
                conv_epoch = row["epoch"]
                break

    run_dir = path.parent
    return {
        "run": str(run_dir.relative_to(ROOT)) if str(run_dir).startswith(str(ROOT)) else str(run_dir),
        "source": read_text(r"SOURCE_NAMES:\s*([A-Za-z0-9_]+)", text),
        "target": read_text(r"TARGET_NAMES:\s*([A-Za-z0-9_]+)", text),
        "id_w": read_float(r"ID_LOSS_WEIGHT:\s*([0-9.]+)", text),
        "i2t_w": read_float(r"I2T_LOSS_WEIGHT:\s*([0-9.]+)", text),
        "kl_w": read_float(r"KL_LOSS_WEIGHT:\s*([0-9.]+)", text),
        "moe_consist_w": read_float(r"MOE_CONSIST_LOSS_WEIGHT:\s*([0-9.]+)", text),
        "da_ins_w": read_float(r"DA_INS_LOSS_WEIGHT:\s*([0-9.]+)", text),
        "mv_recon_w": read_float(r"MV_RECON_LOSS_WEIGHT:\s*([0-9.]+)", text),
        "cls_ins_w": read_float(r"CLS_INS_LOSS_WEIGHT:\s*([0-9.]+)", text),
        "ins_dis_w": read_float(r"INS_DIS_LOSS_WEIGHT:\s*([0-9.]+)", text),
        "best_epoch": best["epoch"],
        "best_mAP": best["mAP"],
        "best_R1": best["R1"],
        "conv_epoch": conv_epoch,
        "avg_speed": round(mean(speeds), 3) if speeds else None,
        "avg_time_per_batch": round(mean(time_per_batch), 4) if time_per_batch else None,
        "num_eval_points": len(evals),
    }


def main():
    args = parse_args()
    results_root = ROOT / args.results_root
    logs = sorted(results_root.rglob("train_log.txt"))
    rows = [summarize_log(path, args.target_ratio) for path in logs]
    rows.sort(key=lambda row: (row["best_mAP"] is None, -(row["best_mAP"] or 0.0), row["run"]))

    if not rows:
        print(f"No train_log.txt files found under {results_root}")
        return

    headers = list(rows[0])
    if args.output_csv:
        output_path = ROOT / args.output_csv
    else:
        output_path = results_root / "ablation_summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")
    print("| run | best mAP | best R1 | conv epoch | speed |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for row in rows:
        print(
            f"| {row['run']} | {row['best_mAP']} | {row['best_R1']} | "
            f"{row['conv_epoch']} | {row['avg_speed']} |"
        )


if __name__ == "__main__":
    main()
