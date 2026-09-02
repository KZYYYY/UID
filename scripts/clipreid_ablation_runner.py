import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parents[1]


STAGE2_ABLATIONS = {
    "baseline": [],
    "no_id": ["MODEL.ID_LOSS_WEIGHT", "0.0"],
    "no_i2t": ["MODEL.I2T_LOSS_WEIGHT", "0.0"],
    "no_kl": ["MODEL.KL_LOSS_WEIGHT", "0.0"],
    "no_text_aux": [
        "MODEL.I2T_LOSS_WEIGHT",
        "0.0",
        "MODEL.KL_LOSS_WEIGHT",
        "0.0",
    ],
}


SCENARIOS = {
    "veri2opri": {
        "source": "veri",
        "target": "opri",
        "weight": "results/veri_mvdc_stride8_bs8_veri2opri_0524_cas_id_loss/ViT-16-based/ViT-B-16_multi_view_70.pth",
    },
    "vehiclex2opri": {
        "source": "vehiclex",
        "target": "opri",
        "weight": "results/vehiclex_mvdc_stride8_bs8_vehiclex2opri_0528_satge1_120_best/ViT-16-based/ViT-B-16_multi_view_90.pth",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Launch CLIP-ReID loss ablation experiments.")
    parser.add_argument("--config-file", default="configs/veri/vit_prom.yml")
    parser.add_argument("--train-script", default="train_clipreid_vehicle_cam.py")
    parser.add_argument("--output-base", default="results/ablation_loss_efficiency")
    parser.add_argument("--cuda-devices", default="0")
    parser.add_argument("--seeds", default="1234")
    parser.add_argument("--scenarios", default=",".join(SCENARIOS))
    parser.add_argument("--experiments", default=",".join(STAGE2_ABLATIONS))
    parser.add_argument("--id-loss-weights", default="0.0,0.25,0.5,1.0,2.0",
                        help="Comma-separated values for MODEL.ID_LOSS_WEIGHT sweeps.")
    parser.add_argument("--i2t-loss-weights", default="0.0,0.25,0.5,1.8,2.0",
                        help="Comma-separated values for MODEL.I2T_LOSS_WEIGHT sweeps.")
    parser.add_argument("--include-sweeps", action="store_true",
                        help="Run weight sweep experiments for ID_LOSS_WEIGHT and KL_LOSS_WEIGHT.")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--eval-period", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="Only print commands.")
    parser.add_argument("--execute", action="store_true", help="Run commands sequentially.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop immediately if one run fails.")
    return parser.parse_args()


def parse_float_list(text: str):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def format_weight_value(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "n")


def build_command(args, scenario_name: str, exp_name: str, seed: str) -> List[str]:
    if scenario_name not in SCENARIOS:
        raise KeyError(f"Unknown scenario: {scenario_name}. Known: {sorted(SCENARIOS)}")
    if exp_name not in STAGE2_ABLATIONS:
        raise KeyError(f"Unknown experiment: {exp_name}. Known: {sorted(STAGE2_ABLATIONS)}")
    scenario = SCENARIOS[scenario_name]
    output_dir = Path(args.output_base) / scenario_name / f"{exp_name}_seed{seed}" / "ViT-16-based"
    opts = [
        "OUTPUT_DIR",
        str(output_dir),
        "SOLVER.SEED",
        str(seed),
        "DATASETS.SOURCE_NAMES",
        scenario["source"],
        "DATASETS.TARGET_NAMES",
        scenario["target"],
        "TEST.WEIGHT",
        scenario["weight"],
        "DATALOADER.NUM_WORKERS",
        str(args.num_workers),
    ]
    if args.max_epochs is not None:
        opts += ["SOLVER.STAGE2.MAX_EPOCHS", str(args.max_epochs)]
    if args.eval_period is not None:
        opts += ["SOLVER.STAGE2.EVAL_PERIOD", str(args.eval_period)]
    opts += STAGE2_ABLATIONS[exp_name]
    return [
        "python",
        args.train_script,
        "--config_file",
        args.config_file,
        *opts,
    ]


def build_sweep_command(args, scenario_name: str, sweep_name: str, weight_name: str, weight_value: float, seed: str) -> List[str]:
    if scenario_name not in SCENARIOS:
        raise KeyError(f"Unknown scenario: {scenario_name}. Known: {sorted(SCENARIOS)}")
    scenario = SCENARIOS[scenario_name]
    weight_key = f"MODEL.{weight_name}"
    weight_suffix = format_weight_value(weight_value)
    output_dir = Path(args.output_base) / scenario_name / f"{sweep_name}_{weight_suffix}_seed{seed}" / "ViT-16-based"
    opts = [
        "OUTPUT_DIR",
        str(output_dir),
        "SOLVER.SEED",
        str(seed),
        "DATASETS.SOURCE_NAMES",
        scenario["source"],
        "DATASETS.TARGET_NAMES",
        scenario["target"],
        "TEST.WEIGHT",
        scenario["weight"],
        "DATALOADER.NUM_WORKERS",
        str(args.num_workers),
        weight_key,
        str(weight_value),
    ]
    if args.max_epochs is not None:
        opts += ["SOLVER.STAGE2.MAX_EPOCHS", str(args.max_epochs)]
    if args.eval_period is not None:
        opts += ["SOLVER.STAGE2.EVAL_PERIOD", str(args.eval_period)]
    return [
        "python",
        args.train_script,
        "--config_file",
        args.config_file,
        *opts,
    ]


def build_sweep_command(args, scenario_name: str, sweep_name: str, weight_name: str, weight_value: float, seed: str) -> List[str]:
    if scenario_name not in SCENARIOS:
        raise KeyError(f"Unknown scenario: {scenario_name}. Known: {sorted(SCENARIOS)}")
    scenario = SCENARIOS[scenario_name]
    weight_key = f"MODEL.{weight_name}"
    weight_suffix = format_weight_value(weight_value)
    output_dir = Path(args.output_base) / scenario_name / f"{sweep_name}_{weight_suffix}_seed{seed}" / "ViT-16-based"
    opts = [
        "OUTPUT_DIR",
        str(output_dir),
        "SOLVER.SEED",
        str(seed),
        "DATASETS.SOURCE_NAMES",
        scenario["source"],
        "DATASETS.TARGET_NAMES",
        scenario["target"],
        "TEST.WEIGHT",
        scenario["weight"],
        "DATALOADER.NUM_WORKERS",
        str(args.num_workers),
        weight_key,
        str(weight_value),
    ]
    if args.max_epochs is not None:
        opts += ["SOLVER.STAGE2.MAX_EPOCHS", str(args.max_epochs)]
    if args.eval_period is not None:
        opts += ["SOLVER.STAGE2.EVAL_PERIOD", str(args.eval_period)]
    return [
        "python",
        args.train_script,
        "--config_file",
        args.config_file,
        *opts,
    ]


def main():
    args = parse_args()
    scenarios = [item.strip() for item in args.scenarios.split(",") if item.strip()]
    experiments = [item.strip() for item in args.experiments.split(",") if item.strip()]
    seeds = [item.strip() for item in args.seeds.split(",") if item.strip()]
    manifest_path = ROOT / args.output_base / "ablation_manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    rows = []
    for scenario_name in scenarios:
        scenario = SCENARIOS.get(scenario_name)
        if scenario is None:
            raise KeyError(f"Unknown scenario: {scenario_name}. Known: {sorted(SCENARIOS)}")
        weight_path = ROOT / scenario["weight"]
        if not weight_path.exists():
            raise FileNotFoundError(f"Stage1 weight for {scenario_name} not found: {weight_path}")
        for exp_name in experiments:
            for seed in seeds:
                cmd = build_command(args, scenario_name, exp_name, seed)
                rows.append(
                    {
                        "created_at": datetime.now().isoformat(timespec="seconds"),
                        "scenario": scenario_name,
                        "source": scenario["source"],
                        "target": scenario["target"],
                        "experiment": exp_name,
                        "seed": seed,
                        "command": cmd,
                        "cuda_devices": args.cuda_devices,
                    }
                )

        if args.include_sweeps:
            id_loss_weights = parse_float_list(args.id_loss_weights)
            i2t_loss_weights = parse_float_list(args.i2t_loss_weights)
            for weight_value in id_loss_weights:
                if weight_value == 1.0:
                    continue
                sweep_name = "id_loss_weight"
                for seed in seeds:
                    exp_name = f"{sweep_name}_{format_weight_value(weight_value)}"
                    cmd = build_sweep_command(args, scenario_name, sweep_name, "ID_LOSS_WEIGHT", weight_value, seed)
                    rows.append(
                        {
                            "created_at": datetime.now().isoformat(timespec="seconds"),
                            "scenario": scenario_name,
                            "source": scenario["source"],
                            "target": scenario["target"],
                            "experiment": exp_name,
                            "seed": seed,
                            "command": cmd,
                            "cuda_devices": args.cuda_devices,
                        }
                    )
            for weight_value in i2t_loss_weights:
                if weight_value == 1.0:
                    continue
                sweep_name = "i2t_loss_weight"
                for seed in seeds:
                    exp_name = f"{sweep_name}_{format_weight_value(weight_value)}"
                    cmd = build_sweep_command(args, scenario_name, sweep_name, "I2T_LOSS_WEIGHT", weight_value, seed)
                    rows.append(
                        {
                            "created_at": datetime.now().isoformat(timespec="seconds"),
                            "scenario": scenario_name,
                            "source": scenario["source"],
                            "target": scenario["target"],
                            "experiment": exp_name,
                            "seed": seed,
                            "command": cmd,
                            "cuda_devices": args.cuda_devices,
                        }
                    )

    with manifest_path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    failed_rows = []
    for row in rows:
        printable = " ".join(row["command"])
        print(
            f"[{row['scenario']} {row['experiment']} seed={row['seed']}] "
            f"CUDA_VISIBLE_DEVICES={args.cuda_devices} {printable}"
        )
        if args.execute:
            result = subprocess.run(row["command"], cwd=ROOT, env=env, check=False)
            if result.returncode != 0:
                row["returncode"] = result.returncode
                failed_rows.append(row)
                print(
                    f"[FAILED] {row['scenario']} {row['experiment']} seed={row['seed']} "
                    f"returncode={result.returncode}"
                )
                if args.stop_on_error:
                    raise subprocess.CalledProcessError(result.returncode, row["command"])
    if not args.execute or args.dry_run:
        print(f"\nManifest written to: {manifest_path}")
    if failed_rows:
        failed_path = ROOT / args.output_base / "failed_runs.jsonl"
        with failed_path.open("a", encoding="utf-8") as fh:
            for row in failed_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\n{len(failed_rows)} run(s) failed. Details written to: {failed_path}")
        if args.stop_on_error:
            sys.exit(1)


if __name__ == "__main__":
    main()
