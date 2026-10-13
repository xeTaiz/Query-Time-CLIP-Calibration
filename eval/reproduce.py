"""Run the released benchmarks and compare their metrics with the paper table."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("scannet", "scannetpp", "lerf_ovs", "3d_ovs")
METHODS = ("sfs", "normlift", "ludvig", "occam", "vala", "opengaussian", "langsplat")
KEYS = ("dataset", "subset", "task", "method", "variant")
METRICS = ("miou", "precision", "recall")
MODULES = {"scannet": "eval.eval_scannet", "scannetpp": "eval.eval_scannetpp", "lerf_ovs": "eval.eval_lerf_ovs", "3d_ovs": "eval.eval_3d_ovs"}


def compare(rows: list[dict], targets: dict, expected_keys: set[tuple], atol: float) -> dict:
    observed = {}
    for row in rows:
        key = tuple(row[name] for name in KEYS)
        if key in observed:
            raise ValueError(f"Duplicate aggregate row: {key}")
        observed[key] = row
    for key, row in list(observed.items()):
        if key[-1] != "raw":
            continue
        calibrated_key = (*key[:-1], "calibrated")
        if calibrated_key in observed:
            calibrated = observed[calibrated_key]
            observed[(*key[:-1], "delta")] = {
                **dict(zip(KEYS, (*key[:-1], "delta"))),
                **{metric: calibrated[metric] - row[metric] for metric in METRICS},
            }
    precision = targets["source"]["precision"]
    comparisons = []
    missing = []
    for target in targets["rows"]:
        key = tuple(target[name] for name in KEYS)
        if key not in expected_keys:
            continue
        actual = observed.get(key)
        if actual is None:
            missing.append(dict(zip(KEYS, key)))
            continue
        for metric in METRICS:
            value = float(actual[metric])
            published = float(target[metric])
            comparisons.append({
                **dict(zip(KEYS, key)), "metric": metric,
                "observed": value, "paper": published,
                "difference_pp": value - published,
                "published_rounding_match": f"{value:.{precision}f}" == f"{published:.{precision}f}",
                "numeric_equal_to_printed_value": value == published,
                "within_absolute_tolerance": abs(value - published) <= atol,
            })
    return {
        "schema": 1, "units": "percent", "paper_source": targets["source"],
        "absolute_tolerance_pp": atol, "expected_rows": len(expected_keys),
        "missing_rows": missing, "compared_metrics": len(comparisons),
        "published_rounding_matches": sum(row["published_rounding_match"] for row in comparisons),
        "numeric_equal_to_printed_values": sum(row["numeric_equal_to_printed_value"] for row in comparisons),
        "maximum_absolute_difference_pp": max((abs(row["difference_pp"]) for row in comparisons), default=None),
        "rows": comparisons,
        "interpretation": "Published values are rounded, not full-precision experimental targets. A rounding match does not establish bitwise equality with historical predictions or counts.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "reproduction")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--tasks", nargs="+", choices=("multiclass", "single_query"), default=["multiclass", "single_query"])
    parser.add_argument("--reduced", action=argparse.BooleanOptionalAction, default=True, help="also run paper ScanNet10 and ScanNet++20 vocabularies")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--assignment-device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--compare-only", action="store_true", help="compare previously generated clean evaluator aggregates without rerunning inference")
    parser.add_argument("--atol", type=float, default=0.0, help="additional absolute comparison tolerance in percentage points; does not alter the published-rounding comparison")
    args = parser.parse_args()
    if args.threads < 1 or args.atol < 0:
        parser.error("threads must be positive and atol nonnegative")
    for values in (args.datasets, args.methods, args.tasks):
        if len(values) != len(set(values)):
            parser.error("datasets, methods and tasks must not contain duplicates")
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = [(dataset, "full") for dataset in args.datasets]
    if args.reduced:
        jobs += [(dataset, subset) for dataset, subset in (("scannet", "10"), ("scannetpp", "20")) if dataset in args.datasets]
    environment = {**os.environ, "OMP_NUM_THREADS": str(args.threads), "MKL_NUM_THREADS": str(args.threads), "OPENBLAS_NUM_THREADS": str(args.threads)}
    rows = []
    failures = []
    for dataset, subset in jobs:
        destination = args.output / dataset / subset
        command = ["uv", "run", "--locked", "--no-default-groups", "--group",
                   "point-runtime" if dataset in {"scannet", "scannetpp"} else "image-runtime",
                   "python", "-m", MODULES[dataset], "--data", str(args.data.resolve()),
                   "--output", str(destination.resolve()), "--methods", *args.methods,
                   "--tasks", *args.tasks, "--device", args.device]
        if dataset in {"scannet", "scannetpp"}:
            command += ["--subset", subset, "--assignment-device", args.assignment_device, "--cache-dir", str((args.output / dataset / "caches").resolve())]
        if not args.compare_only:
            point_runtime = dataset in {"scannet", "scannetpp"}
            child_environment = environment.copy()
            child_environment["UV_PROJECT_ENVIRONMENT"] = str(ROOT / (".venv" if point_runtime else ".venv-image"))
            cuda_home = child_environment.get("CUDA_HOME_POINT" if point_runtime else "CUDA_HOME_IMAGE") or child_environment.get("CUDA_HOME")
            if cuda_home:
                child_environment["CUDA_HOME"] = cuda_home
                child_environment["PATH"] = str(Path(cuda_home) / "bin") + os.pathsep + child_environment.get("PATH", "")
            if child_environment.get("TORCH_EXTENSIONS_DIR"):
                child_environment["TORCH_EXTENSIONS_DIR"] = str(Path(child_environment["TORCH_EXTENSIONS_DIR"]) / ("point-runtime" if point_runtime else "image-runtime"))
            print("Running", " ".join(command), flush=True)
            result = subprocess.run(command, cwd=ROOT, env=child_environment)
            if result.returncode:
                failures.append({"dataset": dataset, "subset": subset, "exit_code": result.returncode})
                continue
        aggregate = destination / "aggregate.json"
        if not aggregate.is_file():
            failures.append({"dataset": dataset, "subset": subset, "error": f"Missing aggregate: {aggregate}"})
            continue
        rows.extend(json.loads(aggregate.read_text())["rows"])
    targets = json.loads(Path(__file__).with_name("paper_targets.json").read_text())
    expected_keys = {
        tuple(row[name] for name in KEYS)
        for row in targets["rows"]
        if (row["dataset"], row["subset"]) in jobs and row["method"] in args.methods and row["task"] in args.tasks
    }
    report = compare(rows, targets, expected_keys, args.atol)
    report["evaluation_failures"] = failures
    report["compare_only"] = args.compare_only
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if report["rows"]:
        with (args.output / "comparison.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(report["rows"][0]))
            writer.writeheader()
            writer.writerows(report["rows"])
    print(json.dumps({key: value for key, value in report.items() if key not in {"rows", "paper_source"}}, indent=2, allow_nan=False))
    if failures or report["missing_rows"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
