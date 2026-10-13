"""Portable alpha-rendered image benchmarks with dataset-specific pixel protocols.

Canonical features and vocabulary are consumed verbatim: export owns normalization.
Metrics/counts are CPU deterministic; semantic rasterization requires CUDA/gsplat.
Case metric arrays use fractions; aggregate JSON/CSV explicitly use percentages.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from calibration import compute_scene_statistics, score_embeddings

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("sfs", "normlift", "ludvig", "occam", "vala", "opengaussian", "langsplat")
SCENES = {
    "lerf_ovs": ("figurines", "ramen", "teatime", "waldo_kitchen"),
    "3d_ovs": ("bed", "bench", "blue_sofa", "covered_desk", "lawn", "office_desk", "room", "snacks", "sofa", "table"),
}
THRESHOLDS = np.asarray([round(-1.0 + 0.01 * i, 2) for i in range(201)], dtype=np.float64)
METRICS = ("miou", "precision", "recall")
SUPPORT_EPSILON = 1e-6


@dataclass
class ImageScene:
    directory: Path
    names: list[str]
    text: torch.Tensor
    class_ids: torch.Tensor
    frames: list[dict[str, Any]]
    methods: dict[str, Any]
    manifest: dict[str, Any]


def _asset(directory: Path, name: str) -> Path:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"canonical asset must be a portable relative path: {name}")
    return directory / path


def _load(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=True)


def _float_tensor(value: Any, shape: tuple[int, ...], label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or tuple(value.shape) != shape:
        raise ValueError(f"{label} must be float32 {shape}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{label} contains nonfinite values")
    return value


def load_scene(data: Path, dataset: str, scene: str) -> ImageScene:
    directory = data / dataset / scene
    manifest = json.loads((directory / "scene.json").read_text())
    if manifest["dataset"] != dataset or manifest["scene"] != scene:
        raise ValueError(f"scene identity mismatch: {directory}")
    vocabulary = _load(_asset(directory, manifest["vocabulary"]))
    names = vocabulary["names"]
    if not names or any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ValueError(f"{directory}: vocabulary must be nonempty and unique")
    ordered_names = sorted(names, key=lambda name: (name.casefold(), name)) if dataset == "lerf_ovs" else sorted(names)
    if names != ordered_names:
        raise ValueError(f"{directory}: vocabulary does not follow the canonical {dataset} ordering")
    text = _float_tensor(vocabulary["features"], (len(names), 512), "vocabulary features")
    class_ids = vocabulary["class_ids"]
    if not isinstance(class_ids, torch.Tensor) or class_ids.dtype != torch.int64 or class_ids.shape != (len(names),):
        raise ValueError("class_ids must be int64 [Q]")
    if len(torch.unique(class_ids)) != len(names):
        raise ValueError("class_ids must be unique")
    frames = _load(_asset(directory, manifest["frames"]))["frames"]
    if not frames or len({frame["filename"] for frame in frames}) != len(frames):
        raise ValueError("frames must be nonempty with unique filenames")
    for frame in frames:
        _asset(directory, frame["filename"])
        height, width = int(frame["height"]), int(frame["width"])
        if min(height, width) <= 0:
            raise ValueError("frame dimensions must be positive")
        _float_tensor(frame["viewmat"], (4, 4), "viewmat")
        _float_tensor(frame["K"], (3, 3), "K")
        masks = frame["masks"]
        if not isinstance(masks, torch.Tensor) or masks.dtype != torch.bool or masks.shape != (len(names), height, width):
            raise ValueError("frame masks must be bool [Q,H,W] in vocabulary order")
    return ImageScene(directory, names, text, class_ids, frames, manifest["methods"], manifest)


def load_method(scene: ImageScene, method: str) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    paths = scene.methods[method]
    geometry = _load(_asset(scene.directory, paths["gaussians"]))
    features = _load(_asset(scene.directory, paths["features"]))
    if not isinstance(features, torch.Tensor) or features.dtype != torch.float32 or features.ndim != 2 or features.shape[1] != 512:
        raise ValueError("canonical features must be float32 [N,512]")
    count = len(features)
    valid = _load(_asset(scene.directory, paths["feature_valid"]))
    if not isinstance(valid, torch.Tensor) or valid.dtype != torch.bool or valid.shape != (count,):
        raise ValueError("feature_valid must be bool [N]")
    for key, width in (("means", 3), ("scales", 3), ("quats", 4)):
        _float_tensor(geometry[key], (count, width), key)
    opacity = geometry["opacity"]
    if opacity.shape == (count, 1):
        opacity = opacity[:, 0]
    geometry["opacity"] = _float_tensor(opacity, (count,), "opacity")
    if bool((geometry["scales"] <= 0).any()) or bool(((opacity < 0) | (opacity > 1)).any()):
        raise ValueError("canonical geometry must contain activated positive scales and [0,1] opacity")
    # No normalization here: source arithmetic (including sub-epsilon rows) is frozen.
    if bool((features[~valid] != 0).any()):
        raise ValueError("adapter-invalid canonical feature rows must be zero")
    return geometry, features, valid


class SemanticRenderer:
    """The authoritative packed gsplat arbitrary-channel alpha composite, without SH."""

    def __init__(self, geometry: Mapping[str, torch.Tensor], device: torch.device):
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("image inference requires a CUDA device and CUDA-compatible gsplat")
        try:
            from gsplat import rasterization
        except ImportError as error:
            raise RuntimeError("image inference requires gsplat (canonical renderer: gsplat 1.5.3)") from error
        self.rasterization = rasterization
        self.device = device
        self.geometry = {key: geometry[key].to(device).contiguous() for key in ("means", "quats", "scales", "opacity")}

    @torch.inference_mode()
    def render(self, values: torch.Tensor, frame: Mapping[str, Any]) -> np.ndarray:
        if values.ndim != 2 or len(values) != len(self.geometry["means"]) or not values.shape[1]:
            raise ValueError("render values must be [N,C], C>0")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("render values must be finite; ineligible scores must be zeroed")
        image, _alpha, _meta = self.rasterization(
            self.geometry["means"], self.geometry["quats"], self.geometry["scales"],
            self.geometry["opacity"], values.to(self.device).contiguous(),
            frame["viewmat"].to(self.device)[None], frame["K"].to(self.device)[None],
            int(frame["width"]), int(frame["height"]), sh_degree=None, packed=True, channel_chunk=32,
        )
        result = image[0].cpu().numpy()
        if result.shape != (frame["height"], frame["width"], values.shape[1]) or not np.isfinite(result).all():
            raise FloatingPointError("invalid semantic rasterization output")
        return result


def confusion_counts(prediction: np.ndarray, truth: np.ndarray, population: np.ndarray | None = None) -> np.ndarray:
    if prediction.dtype != bool or truth.dtype != bool or prediction.shape != truth.shape:
        raise ValueError("prediction and truth must be aligned boolean arrays")
    if population is None:
        population = np.ones(truth.shape, dtype=bool)
    if population.dtype != bool or population.shape != truth.shape:
        raise ValueError("population must be an aligned boolean array")
    predicted, actual = prediction & population, truth & population
    tp = np.count_nonzero(predicted & actual)
    fp = np.count_nonzero(predicted & ~actual)
    fn = np.count_nonzero(~predicted & actual)
    tn = np.count_nonzero(population) - tp - fp - fn
    return np.asarray((tp, fp, fn, tn), dtype=np.int64)


def count_metrics(counts: np.ndarray) -> np.ndarray:
    """Return (...,3) fractional IoU, precision, recall; empty denominators are zero."""
    counts = np.asarray(counts, dtype=np.int64)
    tp, fp, fn = counts[..., 0], counts[..., 1], counts[..., 2]
    denominators = np.stack((tp + fp + fn, tp + fp, tp + fn), axis=-1)
    return np.divide(tp[..., None], denominators, out=np.zeros(denominators.shape, dtype=np.float64), where=denominators != 0)


def multiclass_counts(dataset: str, labels: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """LERF ignores ambiguous/background pixels; OVS includes every class and pixel."""
    population = masks.sum(axis=0) == 1 if dataset == "lerf_ovs" else np.ones(labels.shape, dtype=bool)
    queries, counts = [], []
    for index, mask in enumerate(masks):
        if dataset == "lerf_ovs" and not (mask & population).any():
            continue
        queries.append(index)
        counts.append(confusion_counts(labels == index, mask, population))
    return np.asarray(queries, dtype=np.int64), np.asarray(counts, dtype=np.int64).reshape(-1, 4)


def binary_curve_counts(dataset: str, scores: np.ndarray, masks: np.ndarray, support: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Full-image negatives, present GT only, >= threshold, unsupported always reject."""
    queries, counts = [], []
    # LERF's original float32 image >= Python-float comparison rounds the scalar
    # to float32. OVS's authoritative sorted-count routine compares in float64.
    thresholds = THRESHOLDS.astype(np.float32) if dataset == "lerf_ovs" else THRESHOLDS
    for index, truth in enumerate(masks):
        positives = int(truth.sum())
        if not positives:
            continue
        column = scores[..., index]
        dtype = np.float32 if dataset == "lerf_ovs" else np.float64
        positive_scores = np.sort(column[support & truth].astype(dtype, copy=False))
        negative_scores = np.sort(column[support & ~truth].astype(dtype, copy=False))
        tp = len(positive_scores) - np.searchsorted(positive_scores, thresholds, side="left")
        fp = len(negative_scores) - np.searchsorted(negative_scores, thresholds, side="left")
        queries.append(index)
        counts.append(np.stack((tp, fp, positives - tp, truth.size - positives - fp), axis=-1))
    return np.asarray(queries, dtype=np.int64), np.asarray(counts, dtype=np.int64).reshape(-1, len(THRESHOLDS), 4)


def scene_macro(dataset: str, task: str, query_indices: np.ndarray, counts: np.ndarray) -> np.ndarray:
    values = count_metrics(counts)
    if not len(values):
        raise ValueError("cannot aggregate a scene with no eligible frame-query pairs")
    if dataset == "3d_ovs" and task == "multiclass":
        return values.mean(axis=0)
    return np.stack([values[query_indices == index].mean(axis=0) for index in np.unique(query_indices)]).mean(axis=0)


def loso(curves: Mapping[str, np.ndarray], scenes: Sequence[str]) -> tuple[np.ndarray, dict[str, Any]]:
    """Tune only other scenes' equal-scene mean mIoU, smallest exact-tie threshold."""
    if set(curves) != set(scenes) or len(scenes) < 2:
        raise ValueError("LOSO requires fresh curves for the complete canonical scene cohort")
    folds, heldout_values = {}, []
    for heldout in scenes:
        training = [scene for scene in scenes if scene != heldout]
        objective = np.stack([curves[scene][:, 0] for scene in training]).mean(axis=0)
        selected = int(np.argmax(objective))
        values = curves[heldout][selected]
        heldout_values.append(values)
        folds[heldout] = {
            "training_scenes": training, "selected_threshold": float(THRESHOLDS[selected]),
            "training_miou_percent": float(objective[selected] * 100),
            "heldout_metrics_percent": dict(zip(METRICS, (values * 100).tolist())),
            "training_miou_curve_percent": (objective * 100).tolist(),
            "tie_break": "smallest threshold among exact objective ties",
        }
    return np.stack(heldout_values).mean(axis=0), folds


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def evaluate_case(scene: ImageScene, dataset: str, method: str, tasks: Sequence[str], device: torch.device, output: Path) -> dict[str, Any]:
    geometry, features, adapter_valid = load_method(scene, method)
    statistics = compute_scene_statistics(features)
    # Image protocols deliberately do NOT exclude low-opacity feature primitives.
    eligible = adapter_valid & statistics.feature_valid
    renderer = SemanticRenderer(geometry, device)
    output.mkdir(parents=True, exist_ok=True)
    support_maps = []
    for frame in scene.frames:
        support_maps.append(renderer.render(eligible.float()[:, None], frame)[..., 0] > SUPPORT_EPSILON)
    inputs = {"scene.json": _sha256(scene.directory / "scene.json")}
    for name in (scene.manifest["vocabulary"], scene.manifest["frames"], *scene.methods[method].values()):
        inputs[name] = _sha256(_asset(scene.directory, name))
    result: dict[str, Any] = {
        "schema": 1, "dataset": dataset, "scene": scene.manifest["scene"], "method": method,
        "metric_units": "fraction in count arrays; percent in summaries", "input_sha256": inputs,
        "names": scene.names, "class_ids": scene.class_ids.tolist(),
        "frames": [frame["filename"] for frame in scene.frames],
        "primitive_count": len(features), "feature_valid_count": statistics.valid_count,
        "eligible_count": int(eligible.sum()), "opacity_threshold": None,
        "support_epsilon": SUPPORT_EPSILON, "rendering": {"sh_degree": None, "packed": True, "channel_chunk": 32, "background": "zero", "divide_by_support": False},
        "parameters": {"lambda": 0.3, "alpha": 0.4, "ridge": 0.001, "csls_k": 1000},
        "results": {},
    }
    summaries: dict[str, Any] = {}
    for task in tasks:
        summaries[task] = {}
        result["results"][task] = {}
        for variant in ("raw", "calibrated"):
            primitive_scores, diagnostics = score_embeddings(
                statistics, eligible, scene.text, mode=task, raw=variant == "raw",
                lambda_=0.3, alpha=0.4, ridge=0.001, csls_k=1000, device=device,
            )
            primitive_scores[~eligible] = 0
            primitive_scores = primitive_scores.to(device)
            frame_indices, query_indices, all_counts = [], [], []
            for frame_index, (frame, support) in enumerate(zip(scene.frames, support_maps)):
                scores = renderer.render(primitive_scores, frame)
                masks = frame["masks"].numpy()
                proof: dict[str, Any] = {"scores": scores, "support": support, "masks": masks, "names": np.asarray(scene.names), "class_ids": scene.class_ids.numpy(), "filename": np.asarray(frame["filename"])}
                if task == "multiclass":
                    labels = scores.argmax(axis=-1).astype(np.int32)
                    if dataset == "lerf_ovs":
                        labels[~support] = -1
                    proof["prediction"] = labels
                    indices, counts = multiclass_counts(dataset, labels, masks)
                else:
                    scores[~support] = -np.inf
                    indices, counts = binary_curve_counts(dataset, scores, masks, support)
                    proof["thresholds"] = THRESHOLDS
                frame_indices.extend([frame_index] * len(indices))
                query_indices.extend(indices.tolist())
                all_counts.append(counts)
                np.savez_compressed(output / f"{task}_{variant}_frame_{frame_index:04d}.npz", **proof)
            counts = np.concatenate(all_counts, axis=0)
            indices_array = np.asarray(query_indices, dtype=np.int64)
            metrics = scene_macro(dataset, task, indices_array, counts)
            summaries[task][variant] = metrics
            counts_file = f"{task}_{variant}_counts.npz"
            np.savez_compressed(output / counts_file, counts=counts, count_columns=np.asarray(("tp", "fp", "fn", "tn")), metrics=count_metrics(counts), metric_columns=np.asarray(METRICS), metric_units=np.asarray("fraction"), frame_indices=np.asarray(frame_indices), query_indices=indices_array, thresholds=THRESHOLDS if task == "single_query" else np.asarray([]))
            result["results"][task][variant] = {"diagnostics": diagnostics, "counts_file": counts_file, "frame_query_pairs": len(counts), "scene_metrics_percent": (metrics * 100).tolist()}
    _json(output / "result.json", result)
    return summaries


def run_dataset_cli(dataset: str) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / dataset)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--scenes", nargs="+", choices=SCENES[dataset], default=list(SCENES[dataset]))
    parser.add_argument("--tasks", nargs="+", choices=("multiclass", "single_query", "both"), default=["both"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    tasks = ["multiclass", "single_query"] if "both" in args.tasks else list(dict.fromkeys(args.tasks))
    scenes = list(dict.fromkeys(args.scenes))
    methods = list(dict.fromkeys(args.methods))
    full = set(scenes) == set(SCENES[dataset])
    if "single_query" in tasks and not full:
        parser.error("single_query requires the full canonical scene cohort for strict LOSO; use --tasks multiclass for partial runs")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("image inference requires CUDA and CUDA-compatible gsplat; CPU metric helpers are available separately")
    # Match archived numeric policy instead of accepting platform-dependent TF32.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    cases: dict[str, dict[str, Any]] = {method: {} for method in methods}
    for scene_name in scenes:
        scene = load_scene(args.data, dataset, scene_name)
        for method in methods:
            cases[method][scene_name] = evaluate_case(scene, dataset, method, tasks, device, args.output / method / scene_name)
    rows, details = [], {}
    for method in methods:
        details[method] = {}
        for task in tasks:
            details[method][task] = {}
            for variant in ("raw", "calibrated"):
                if task == "single_query":
                    metrics, folds = loso({scene: cases[method][scene][task][variant] for scene in scenes}, SCENES[dataset])
                    details[method][task][variant] = {"folds": folds, "threshold_grid": THRESHOLDS.tolist()}
                else:
                    metrics = np.stack([cases[method][scene][task][variant] for scene in scenes]).mean(axis=0)
                    details[method][task][variant] = {"per_scene_metrics_percent": {scene: dict(zip(METRICS, (cases[method][scene][task][variant] * 100).tolist())) for scene in scenes}}
                rows.append({"dataset": dataset, "subset": "full" if full else "partial", "task": task, "method": method, "variant": variant, **dict(zip(METRICS, (metrics * 100).tolist()))})
    _json(args.output / "aggregate.json", {"schema": 1, "metric_units": "percent", "dataset": dataset, "subset": "full" if full else "partial", "scenes": scenes, "methods": methods, "rows": rows, "details": details, "calibration_sha256": _sha256(ROOT / "calibration.py"), "evaluator_sha256": _sha256(Path(__file__))})
    with (args.output / "aggregate.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("dataset", "subset", "task", "method", "variant", *METRICS))
        writer.writeheader()
        writer.writerows(rows)
