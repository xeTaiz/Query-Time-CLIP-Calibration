"""Full-cohort point evaluation using canonical exports and exact P3 assignment."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

import calibration
from calibration import (
    compute_scene_statistics,
    eligibility_mask,
    load_features,
    load_gaussians,
    load_reference,
    score_embeddings,
)
from . import evaluation
from .evaluation import (
    DatasetVocabularySpec,
    _hash_tensor,
    _safe_mapping,
    _selected_vocabulary,
    _validate_vocabulary,
    evaluate_scores,
    load_or_create_assignment_cache,
)

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("sfs", "normlift", "ludvig", "occam", "vala", "opengaussian", "langsplat")
SCENES = {
    "scannet": (
        "scene0000_00", "scene0062_00", "scene0070_00", "scene0097_00", "scene0140_00",
        "scene0200_00", "scene0347_00", "scene0400_00", "scene0590_00", "scene0645_00",
    ),
    "scannetpp": (
        "09c1414f1b", "0d2ee665be", "27dd4da69e", "3864514494", "3db0a1c8f3", "578511c8a9",
        "5942004064", "9071e139d9", "c50d2d1d42", "d755b3d9d8", "e7af285f7d", "f9f95681fd",
    ),
}
CALIBRATED_THRESHOLDS = {
    "scannet": {
        "sfs": -0.040, "normlift": -0.040, "ludvig": -0.040, "occam": -0.040,
        "vala": -0.040, "opengaussian": -0.024, "langsplat": -0.100,
    },
    "scannetpp": {
        "sfs": -0.020, "normlift": -0.022, "ludvig": -0.020, "occam": -0.017,
        "vala": -0.008, "opengaussian": -0.020, "langsplat": -0.040,
    },
}


def paper_threshold(dataset: str, method: str, raw: bool) -> float:
    if raw:
        return 0.245 if dataset == "scannetpp" else (0.15 if method == "langsplat" else 0.232)
    return CALIBRATED_THRESHOLDS[dataset][method]


def _atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _tensor_identity(name: str, value: torch.Tensor) -> str:
    digest = hashlib.sha256()
    _hash_tensor(digest, name, value)
    return digest.hexdigest()


def _content_identity(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def aggregate_results(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Equal-scene means, never pooled point counts or historical result values."""
    groups: dict[tuple[str, str, str, str, str], list[Mapping[str, Any]]] = {}
    for case in cases:
        key = tuple(str(case[field]) for field in ("dataset", "subset", "method", "task", "variant"))
        groups.setdefault(key, []).append(case)
    output = []
    for key, rows in groups.items():
        scenes = [str(row["scene"]) for row in rows]
        if len(set(scenes)) != len(scenes):
            raise ValueError(f"duplicate scene in aggregate group {key}")
        item = dict(zip(("dataset", "subset", "method", "task", "variant"), key))
        item.update({
            "scene_count": len(rows), "scenes": scenes,
            "aggregation": "equal-scene mean of GT-present-class macro metrics; percentages",
            **{metric: math.fsum(float(row[metric]) for row in rows) / len(rows)
               for metric in ("miou", "precision", "recall")},
        })
        per_class: dict[int, list[Mapping[str, Any]]] = {}
        for row in rows:
            for class_row in row["per_class"]:
                per_class.setdefault(int(class_row["class_id"]), []).append(class_row)
        item["per_class"] = [
            {
                "class_id": class_id, "class_name": values[0]["class_name"],
                "scene_count": len(values),
                **{metric: math.fsum(float(value[metric]) for value in values) / len(values)
                   for metric in ("iou", "precision", "recall")},
                "total_tp": sum(int(value["tp"]) for value in values),
                "total_fp": sum(int(value["fp"]) for value in values),
                "total_fn": sum(int(value["fn"]) for value in values),
                "aggregation": "equal mean over scenes where original GT contains this class; counts are provenance only",
            }
            for class_id, values in sorted(per_class.items())
        ]
        output.append(item)
    return output


def _write_csv(aggregates: Sequence[Mapping[str, Any]], path: Path) -> None:
    fields = ("dataset", "subset", "method", "task", "variant", "scene_count", "miou", "precision", "recall")
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(aggregates)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parser(spec: DatasetVocabularySpec) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Reproduce the full {spec.dataset} point cohort.")
    parser.add_argument("--data", type=Path, default=ROOT / "data", help="canonical exported data root")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / spec.dataset)
    parser.add_argument("--cache-dir", type=Path, default=None, help="shared identity-checked correspondence cache root; defaults to OUTPUT/caches")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--scenes", nargs="+", choices=SCENES[spec.dataset], default=list(SCENES[spec.dataset]))
    parser.add_argument("--subset", choices=("full", *spec.subsets), default="full")
    parser.add_argument("--tasks", nargs="+", choices=("multiclass", "single_query"), default=["multiclass", "single_query"])
    parser.add_argument("--device", default="cuda", help="scoring device")
    parser.add_argument("--assignment-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--ridge", type=float, default=0.001)
    parser.add_argument("--csls-k", type=int, default=1000)
    parser.add_argument("--opacity", "--opacity-threshold", dest="opacity_threshold", type=float, default=0.1)
    parser.add_argument("--raw-threshold", type=float, default=None, help="override paper single-query raw threshold")
    parser.add_argument("--calibrated-threshold", type=float, default=None, help="override paper single-query calibrated threshold")
    parser.add_argument("--chunk-size", type=int, default=16384, help="scoring chunk size")
    parser.add_argument("--point-chunk", type=int, default=2048)
    parser.add_argument("--primitive-chunk", type=int, default=65536)
    return parser


def _environment() -> dict[str, Any]:
    versions = {}
    for package in ("torch", "numpy", "scipy", "open_clip_torch"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    sources = (Path(__file__), Path(evaluation.__file__), Path(calibration.__file__))
    return {
        "python": sys.version, "platform": platform.platform(), "packages": versions,
        "cuda_runtime": torch.version.cuda,
        "cuda_devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        "cpu_workers": os.environ.get("P3_CPU_WORKERS", "4"),
        "source_sha256": {source.name: hashlib.sha256(source.read_bytes()).hexdigest() for source in sources},
        "numeric_policy": {
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "torch_cpu_threads": torch.get_num_threads(),
        },
    }


def _evaluate_case(
    args: argparse.Namespace, spec: DatasetVocabularySpec, scene_name: str, method: str,
    gt: Mapping[str, Any], gt_identity: str, queries: torch.Tensor, candidate_ids: torch.Tensor,
    candidate_names: list[str], reference: torch.Tensor, shared_identity: Mapping[str, Any],
) -> list[dict[str, Any]]:
    case_dir = args.data / spec.dataset / scene_name / method
    gaussians = load_gaussians(case_dir / "gaussians.pt")
    features = load_features(case_dir / "features.pt")
    # One feature reduction and one exact correspondence, shared by all four evaluations.
    statistics = compute_scene_statistics(features, chunk_size=args.chunk_size)
    eligible = eligibility_mask(gaussians, features, opacity_threshold=args.opacity_threshold)
    assignments, cache = load_or_create_assignment_cache(
        args.cache_dir / method / f"{scene_name}.pt", gt["points"], gaussians, eligible,
        device=args.assignment_device, point_chunk=args.point_chunk, primitive_chunk=args.primitive_chunk,
    )
    inputs = {
        **shared_identity, "gt": gt_identity, "features": _tensor_identity("features", features),
        "assignment": cache["identity"], "case_path": str(case_dir),
    }
    rows = []
    names_by_id = dict(zip(candidate_ids.tolist(), candidate_names))
    for raw in (True, False):
        raw_scores = None
        raw_diagnostics = None
        for task in args.tasks:
            # Raw cosine scores do not depend on task; avoid recomputing their matrix.
            if raw and raw_scores is not None:
                scores, diagnostics = raw_scores, raw_diagnostics
            else:
                # Appendix ScanNet10/ScanNet++20 uses the frozen direct-CSLS protocol,
                # unlike the main full-vocabulary clustered multiclass tables.
                appendix_multiclass = (
                    not raw and task == "multiclass"
                    and (spec.dataset, args.subset) in {("scannet", "10"), ("scannetpp", "20")}
                )
                scores, diagnostics = score_embeddings(
                    statistics, eligible, queries, reference=reference,
                    lambda_=args.lambda_, alpha=args.alpha, ridge=args.ridge, csls_k=args.csls_k,
                    mode=task, raw=raw, device=args.device, chunk_size=args.chunk_size,
                    cluster=not appendix_multiclass, csls=True if appendix_multiclass else None,
                )
                if raw:
                    raw_scores, raw_diagnostics = scores, diagnostics
            override = args.raw_threshold if raw else args.calibrated_threshold
            threshold = (paper_threshold(spec.dataset, method, raw) if override is None else override) if task == "single_query" else None
            parameters = {
                "lambda": args.lambda_, "alpha": args.alpha, "ridge": args.ridge,
                "csls_k": args.csls_k, "opacity_threshold": args.opacity_threshold,
                "raw": raw, "mode": task, "fit": "none" if raw else "reference",
                "cluster": diagnostics["scoring"]["cluster"],
                "csls": diagnostics["scoring"]["csls"], "chunk_size": args.chunk_size,
            }
            bundle = {"scores": scores, "eligible": eligible, "metadata": {
                "mode": task, "class_ids": candidate_ids.tolist(),
                "opacity_threshold": args.opacity_threshold, "parameters": parameters,
                "feature_valid": statistics.valid_count,
                "statistics_population": "all finite exact-nonzero rows, including low-opacity rows",
                **diagnostics,
            }}
            result = evaluate_scores(
                gaussians, features, bundle, gt["points"], gt["labels"], candidate_ids,
                mode=task, threshold=threshold, opacity_threshold=args.opacity_threshold,
                assignments=assignments, device=args.assignment_device,
            )
            result.pop("assignments")
            for row in result["per_class"]:
                row["class_name"] = names_by_id[row["class_id"]]
            variant = "raw" if raw else "calibrated"
            result.update({
                "dataset": spec.dataset, "scene": scene_name, "method": method,
                "subset": args.subset, "task": task, "variant": variant,
            })
            result["provenance"]["assignment"]["cache"] = cache
            result["provenance"]["inputs"] = inputs
            result["provenance"]["vocabulary"] = {
                "class_ids": candidate_ids.tolist(), "class_names": candidate_names,
                "subset": args.subset, "binary_negatives": "all original nonignored GT labels, including noncandidate labels",
            }
            result["identity"] = _content_identity({
                "inputs": inputs, "parameters": parameters, "threshold": threshold,
                "dataset": spec.dataset, "scene": scene_name, "method": method,
                "subset": args.subset, "candidate_class_ids": candidate_ids.tolist(),
                "assignment_device": args.assignment_device, "device": args.device,
            })
            case_output = args.output / "cases" / args.subset / task / variant / method / f"{scene_name}.json"
            result["output"] = str(case_output)
            _atomic_json(result, case_output)
            rows.append(result)
            print(json.dumps({key: result[key] for key in ("dataset", "scene", "method", "task", "variant", "miou", "precision", "recall", "output")}), flush=True)
            del result, bundle, scores, diagnostics
        del raw_scores, raw_diagnostics
    # Only small JSON results escape this function; all case-sized tensors are released.
    return rows


def run_dataset_cli(spec: DatasetVocabularySpec, argv: Sequence[str] | None = None) -> None:
    args = _parser(spec).parse_args(argv)
    args.data = args.data.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.cache_dir = (args.output / "caches" if args.cache_dir is None else args.cache_dir.expanduser().resolve())
    for field in ("methods", "scenes", "tasks"):
        if len(set(getattr(args, field))) != len(getattr(args, field)):
            raise ValueError(f"--{field} must not contain duplicates")
    if not 0.0 <= args.opacity_threshold <= 1.0 or not math.isfinite(args.opacity_threshold):
        raise ValueError("--opacity must be finite and in [0,1]")
    for field in ("lambda_", "alpha", "ridge", "raw_threshold", "calibrated_threshold"):
        value = getattr(args, field)
        if value is not None and not math.isfinite(value):
            raise ValueError(f"{field} must be finite")
    if min(args.csls_k, args.chunk_size, args.point_chunk, args.primitive_chunk) <= 0:
        raise ValueError("chunk sizes and --csls-k must be positive")
    # Match the archived campaign and the image evaluator's numeric policy.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    vocabulary_path = args.data / spec.dataset / "vocabulary.pt"
    vocabulary = _safe_mapping(vocabulary_path, "vocabulary")
    full_ids, names, full_queries, subsets = _validate_vocabulary(vocabulary, spec)
    candidate_ids, candidate_names = _selected_vocabulary(full_ids, names, subsets, args.subset)
    positions = {class_id: index for index, class_id in enumerate(full_ids.tolist())}
    queries = full_queries[[positions[class_id] for class_id in candidate_ids.tolist()]]
    reference = load_reference()
    # Fail before an expensive cohort if any requested canonical input is missing.
    for scene in args.scenes:
        paths = [args.data / spec.dataset / scene / "gt.pt"]
        for method in args.methods:
            paths.extend(args.data / spec.dataset / scene / method / name for name in ("gaussians.pt", "features.pt"))
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
    environment = _environment()
    shared_identity = {
        "vocabulary": _tensor_identity("full_queries", full_queries),
        "reference": _tensor_identity("reference", reference),
        "reference_bank": calibration.REFERENCE_BANK_ID,
        "source_sha256": environment["source_sha256"], "packages": environment["packages"],
    }
    run = {
        "schema": 1, "dataset": spec.dataset, "started": datetime.now(timezone.utc).isoformat(),
        "completed": False, "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "environment": environment, "inputs": shared_identity,
        "aggregation": "equal-scene dataset mean of original GT-present requested-class macro metrics; percentages",
        "result_policy": "recompute every result; reuse only content-identity-checked correspondence caches",
    }
    _atomic_json(run, args.output / "run.json")
    cases = []
    for scene in args.scenes:
        gt = _safe_mapping(args.data / spec.dataset / scene / "gt.pt", "ground truth")
        if set(gt) != {"points", "labels"}:
            raise ValueError("gt.pt must contain exactly points and labels")
        digest = hashlib.sha256()
        _hash_tensor(digest, "points", gt["points"])
        _hash_tensor(digest, "labels", gt["labels"])
        gt_identity = digest.hexdigest()
        for method in args.methods:
            cases.extend(_evaluate_case(args, spec, scene, method, gt, gt_identity, queries, candidate_ids, candidate_names, reference, shared_identity))
        del gt
    aggregates = aggregate_results(cases)
    _atomic_json({"schema": 1, "dataset": spec.dataset, "subset": args.subset, "rows": aggregates, "cases": cases}, args.output / "aggregate.json")
    _write_csv(aggregates, args.output / "aggregate.csv")
    run.update({"completed": True, "finished": datetime.now(timezone.utc).isoformat(), "case_count": len(cases)})
    _atomic_json(run, args.output / "run.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Select a portable point evaluation dataset.", add_help=False)
    parser.add_argument("--dataset", choices=tuple(SCENES), required=True)
    args, remaining = parser.parse_known_args()
    if args.dataset == "scannet":
        from .eval_scannet import SPEC
    else:
        from .eval_scannetpp import SPEC
    run_dataset_cli(SPEC, remaining)


if __name__ == "__main__":
    main()
