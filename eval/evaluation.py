"""Exact P3 correspondence and point-level metrics for calibrated CLIP scores."""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
import torch

from calibration import eligibility_mask

_ASSIGNMENT_ALGORITHM = "exact-certified-mahalanobis-float64-wxyz-scalefloor1e-8-v2"
_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}


@dataclass(frozen=True)
class DatasetVocabularySpec:
    """The immutable candidate IDs and named subsets released for one dataset."""

    dataset: str
    class_ids: tuple[int, ...]
    names: tuple[str, ...]
    subsets: Mapping[str, tuple[int, ...]]


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _integer_vector(value: Any, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tensor.ndim != 1 or tensor.dtype not in _INTEGER_DTYPES:
        raise ValueError(f"{name} must be a one-dimensional integer tensor")
    return tensor.to(device="cpu", dtype=torch.int64).contiguous()


def _geometry(gaussians: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not isinstance(gaussians, Mapping):
        raise TypeError("gaussians must be a tensor mapping")
    missing = {"means", "scales", "quats"} - set(gaussians)
    if missing:
        raise ValueError(f"gaussians is missing geometry keys: {sorted(missing)}")
    means = torch.as_tensor(gaussians["means"])
    scales = torch.as_tensor(gaussians["scales"])
    quats = torch.as_tensor(gaussians["quats"])
    if means.ndim != 2 or means.shape[1] != 3:
        raise ValueError(f"gaussian means must have shape [N,3], found {tuple(means.shape)}")
    if scales.shape != means.shape:
        raise ValueError(f"gaussian scales must have shape {tuple(means.shape)}, found {tuple(scales.shape)}")
    if quats.shape != (means.shape[0], 4):
        raise ValueError(f"gaussian quats must have shape [{means.shape[0]},4], found {tuple(quats.shape)}")
    if not (means.is_floating_point() and scales.is_floating_point() and quats.is_floating_point()):
        raise ValueError("gaussian geometry tensors must be floating point")
    return means.detach().cpu(), scales.detach().cpu(), quats.detach().cpu()


def _rotations_wxyz(quats: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quats.unbind(dim=1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        dim=1,
    ).reshape(-1, 3, 3)


def _pair_distances(
    points: np.ndarray,
    indices: np.ndarray,
    means: np.ndarray,
    rotations: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    delta = points - means[indices]
    local = np.einsum(
        "...ji,...j->...i", rotations[indices], delta, optimize=True
    )
    return np.square(local / scales[indices]).sum(axis=-1)


def _nearest_centre_seed(
    points: np.ndarray,
    means: np.ndarray,
    rotations: np.ndarray,
    scales: np.ndarray,
    workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    # Duplicate centres share zero distance; retain their lowest candidate index.
    unique_means, first = np.unique(means, axis=0, return_index=True)
    tree = cKDTree(unique_means)
    _, nearest = tree.query(points, k=1, workers=workers)
    winners = first[np.asarray(nearest, dtype=np.int64)]
    best = _pair_distances(points, winners, means, rotations, scales)
    return winners, best


def _assign_certified_cpu(
    points: np.ndarray,
    means: np.ndarray,
    rotations: np.ndarray,
    scales: np.ndarray,
    *,
    point_chunk: int,
    workers: int,
) -> np.ndarray:
    """Exact scale-binned search; Euclidean queries only establish lower bounds."""

    winners, best = _nearest_centre_seed(
        points, means, rotations, scales, workers
    )
    scale_bins = np.floor(np.log2(scales.max(axis=1))).astype(np.int64)
    for scale_bin in sorted(np.unique(scale_bins), reverse=True):
        members = np.flatnonzero(scale_bins == scale_bin)
        tree = cKDTree(means[members])
        # d_M^2 >= d_E^2 / max_scale^2 for every Gaussian in this bin.
        max_scale = float(scales[members].max()) * (1.0 + 1e-12)
        pending = np.flatnonzero(best > 0)
        k = min(8, len(members))
        while len(pending):
            next_pending: list[np.ndarray] = []
            step = min(point_chunk, max(1, 262_144 // k))
            for offset in range(0, len(pending), step):
                at = pending[offset : offset + step]
                euclidean, local = tree.query(
                    points[at], k=k, workers=workers
                )
                euclidean = np.asarray(euclidean).reshape(len(at), k)
                local = np.asarray(local).reshape(len(at), k)
                candidate_ids = members[local]
                distance = _pair_distances(
                    points[at, None, :],
                    candidate_ids,
                    means,
                    rotations,
                    scales,
                )
                values = distance.min(axis=1)
                choices = np.where(
                    distance == values[:, None],
                    candidate_ids,
                    len(means),
                ).min(axis=1)
                improve = (values < best[at]) | (
                    (values == best[at]) & (choices < winners[at])
                )
                changed = at[improve]
                best[changed] = values[improve]
                winners[changed] = choices[improve]
                if k < len(members):
                    lower_bound = np.square(euclidean[:, -1] / max_scale)
                    uncertain = lower_bound <= best[at] * (1.0 + 1e-12)
                    next_pending.append(at[uncertain])
            pending = (
                np.concatenate(next_pending)
                if next_pending
                else np.empty(0, dtype=np.int64)
            )
            k = min(k * 2, len(members))
    if not np.isfinite(best).all():
        raise FloatingPointError(
            "Mahalanobis distance overflowed for at least one point"
        )
    return winners


def assign_mahalanobis(
    points: Any,
    gaussians: Mapping[str, Any],
    eligible: Any,
    *,
    device: str | torch.device = "cuda",
    point_chunk: int = 2_048,
    primitive_chunk: int = 65_536,
) -> torch.Tensor:
    """Assign every point to its exact minimum-Mahalanobis eligible Gaussian.

    CPU search expands scale-binned Euclidean queries only until a conservative
    Mahalanobis lower bound certifies the winner; it is exact, not fixed-k. CUDA
    first resolves exact zero-centre matches, then evaluates every candidate for
    every unresolved point in float64 blocks. Returned indices address the
    original Gaussian arrays. With no eligible Gaussian every assignment is -1.
    """

    point_chunk = _positive_integer(point_chunk, "point_chunk")
    primitive_chunk = _positive_integer(primitive_chunk, "primitive_chunk")
    means, scales, quats = _geometry(gaussians)
    points_cpu = torch.as_tensor(points)
    if points_cpu.ndim != 2 or points_cpu.shape[1] != 3 or not points_cpu.is_floating_point():
        raise ValueError(f"points must be a floating tensor with shape [M,3], found {tuple(points_cpu.shape)}")
    points_cpu = points_cpu.detach().cpu()
    if not torch.isfinite(points_cpu).all():
        raise ValueError("points contain nonfinite coordinates")

    eligible_cpu = torch.as_tensor(eligible)
    if eligible_cpu.shape != (means.shape[0],) or eligible_cpu.dtype != torch.bool:
        raise ValueError(f"eligible must be bool[{means.shape[0]}], found {tuple(eligible_cpu.shape)} {eligible_cpu.dtype}")
    eligible_cpu = eligible_cpu.detach().cpu().contiguous()
    candidate_indices = torch.nonzero(eligible_cpu, as_tuple=False).flatten()
    if candidate_indices.numel() == 0:
        return torch.full((points_cpu.shape[0],), -1, dtype=torch.int64)
    if points_cpu.shape[0] == 0:
        return torch.empty(0, dtype=torch.int64)

    selected_means = means.index_select(0, candidate_indices).to(dtype=torch.float64)
    selected_scales = scales.index_select(0, candidate_indices).to(dtype=torch.float64)
    selected_quats = quats.index_select(0, candidate_indices).to(dtype=torch.float64)
    if not (
        torch.isfinite(selected_means).all()
        and torch.isfinite(selected_scales).all()
        and torch.isfinite(selected_quats).all()
    ):
        raise ValueError("eligible Gaussian geometry contains nonfinite values")
    if not (selected_scales > 0).all():
        raise ValueError("eligible Gaussian scales must be strictly positive activated standard deviations")
    selected_scales = selected_scales.clamp_min(1e-8)
    quat_norms = torch.linalg.vector_norm(selected_quats, dim=1, keepdim=True)
    if not (quat_norms > 0).all():
        raise ValueError("eligible WXYZ quaternions must have nonzero norm")
    selected_quats = selected_quats / quat_norms

    compute_device = torch.device(device)
    if compute_device.type not in {"cpu", "cuda"}:
        raise ValueError("assignment device must be 'cpu' or 'cuda'")
    if compute_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA assignment requested but CUDA is unavailable")

    points_np = points_cpu.to(dtype=torch.float64).contiguous().numpy()
    means_np = selected_means.contiguous().numpy()
    scales_np = selected_scales.contiguous().numpy()
    rotations_cpu = _rotations_wxyz(selected_quats)
    rotations_np = rotations_cpu.contiguous().numpy()
    workers = max(1, int(os.environ.get("P3_CPU_WORKERS", "4")))

    if compute_device.type == "cpu":
        local_winners = _assign_certified_cpu(
            points_np,
            means_np,
            rotations_np,
            scales_np,
            point_chunk=point_chunk,
            workers=workers,
        )
        return candidate_indices.index_select(
            0, torch.from_numpy(local_winners).to(dtype=torch.int64)
        )
    seeded_winners, seeded_best = _nearest_centre_seed(
        points_np, means_np, rotations_np, scales_np, workers
    )

    output = candidate_indices.index_select(
        0, torch.from_numpy(seeded_winners).to(dtype=torch.int64)
    )
    unresolved = np.flatnonzero(seeded_best > 0)
    if not len(unresolved):
        return output

    selected_means = selected_means.to(compute_device)
    selected_scales = selected_scales.to(compute_device)
    rotations = rotations_cpu.to(compute_device)
    inverse_variances = selected_scales.square().reciprocal()
    precision = torch.matmul(
        rotations * inverse_variances[:, None, :], rotations.transpose(1, 2)
    )
    linear = -2.0 * torch.matmul(precision, selected_means[:, :, None]).squeeze(2)
    constant = (
        selected_means
        * torch.matmul(precision, selected_means[:, :, None]).squeeze(2)
    ).sum(dim=1, keepdim=True)
    coefficients = torch.cat(
        (
            precision[:, (0, 1, 2), (0, 1, 2)],
            2.0 * precision[:, (0, 0, 1), (1, 2, 2)],
            linear,
            constant,
        ),
        dim=1,
    )
    if not torch.isfinite(coefficients).all():
        raise ValueError("eligible Gaussian covariance expansion is nonfinite")

    unresolved_tensor = torch.from_numpy(unresolved).to(dtype=torch.int64)
    with torch.inference_mode():
        for point_start in range(0, unresolved_tensor.numel(), point_chunk):
            point_stop = min(
                point_start + point_chunk, unresolved_tensor.numel()
            )
            point_indices = unresolved_tensor[point_start:point_stop]
            point = points_cpu.index_select(0, point_indices).to(
                device=compute_device, dtype=torch.float64
            )
            x, y, z = point.unbind(dim=1)
            basis = torch.stack(
                (
                    x * x,
                    y * y,
                    z * z,
                    x * y,
                    x * z,
                    y * z,
                    x,
                    y,
                    z,
                    torch.ones_like(x),
                ),
                dim=1,
            )
            best = torch.full(
                (point.shape[0],), torch.inf, dtype=torch.float64, device=compute_device
            )
            winner = torch.zeros(point.shape[0], dtype=torch.int64, device=compute_device)
            for primitive_start in range(0, coefficients.shape[0], primitive_chunk):
                primitive_stop = min(
                    primitive_start + primitive_chunk, coefficients.shape[0]
                )
                distance = basis @ coefficients[primitive_start:primitive_stop].T
                block_best, block_winner = distance.min(dim=1)
                improve = block_best < best
                best[improve] = block_best[improve]
                winner[improve] = block_winner[improve] + primitive_start
            if not torch.isfinite(best).all():
                raise FloatingPointError("Mahalanobis distance overflowed for at least one point")
            output[point_indices] = candidate_indices.index_select(
                0, winner.cpu()
            )
    return output


def _hash_tensor(digest: Any, name: str, value: Any) -> None:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest.update(name.encode("utf-8"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(str(tensor.dtype).encode("ascii"))
    if tensor.numel():
        digest.update(memoryview(tensor.numpy()).cast("B"))


def assignment_identity(points: Any, gaussians: Mapping[str, Any], eligible: Any) -> str:
    """Return the content identity used by safe assignment caches."""

    means, scales, quats = _geometry(gaussians)
    digest = hashlib.sha256(_ASSIGNMENT_ALGORITHM.encode("ascii"))
    _hash_tensor(digest, "points", points)
    _hash_tensor(digest, "means", means)
    _hash_tensor(digest, "scales", scales)
    _hash_tensor(digest, "quats", quats)
    _hash_tensor(digest, "eligible", eligible)
    return digest.hexdigest()


def _atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_or_create_assignment_cache(
    path: str | Path,
    points: Any,
    gaussians: Mapping[str, Any],
    eligible: Any,
    *,
    device: str | torch.device = "cuda",
    point_chunk: int = 2_048,
    primitive_chunk: int = 65_536,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Load an identity-matched assignment or atomically create it."""

    cache_path = Path(path)
    identity = assignment_identity(points, gaussians, eligible)
    if cache_path.is_file():
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        if not isinstance(cached, dict) or set(cached) != {"assignments", "metadata"}:
            raise ValueError(f"invalid assignment cache structure: {cache_path}")
        metadata = cached["metadata"]
        if not isinstance(metadata, dict) or metadata.get("identity") != identity:
            raise ValueError(
                f"assignment cache identity mismatch: {cache_path}; use a case-specific cache path"
            )
        assignments = _validate_assignments(
            cached["assignments"],
            torch.as_tensor(points).shape[0],
            _geometry(gaussians)[0].shape[0],
            torch.as_tensor(eligible).detach().cpu(),
        )
        return assignments, {**metadata, "cache_hit": True, "path": str(cache_path)}

    assignments = assign_mahalanobis(
        points,
        gaussians,
        eligible,
        device=device,
        point_chunk=point_chunk,
        primitive_chunk=primitive_chunk,
    )
    metadata = {
        "schema": 1,
        "identity": identity,
        "algorithm": _ASSIGNMENT_ALGORITHM,
        "points": int(assignments.numel()),
        "eligible_gaussians": int(torch.as_tensor(eligible, dtype=torch.bool).sum().item()),
    }
    _atomic_torch_save(
        {"assignments": assignments, "metadata": metadata}, cache_path
    )
    return assignments, {**metadata, "cache_hit": False, "path": str(cache_path)}


def _candidate_ids_from_metadata(metadata: Any) -> torch.Tensor | None:
    if not isinstance(metadata, Mapping):
        return None
    found: list[torch.Tensor] = []
    for key in ("class_ids", "candidate_class_ids", "query_class_ids"):
        if key in metadata:
            found.append(_integer_vector(metadata[key], f"score metadata {key}"))
    if not found:
        return None
    if any(not torch.equal(found[0], value) for value in found[1:]):
        raise ValueError("score metadata contains conflicting candidate-column mappings")
    return found[0]


def _prepare_scores(
    gaussians: Mapping[str, Any],
    features: Any,
    score_input: Any,
    class_ids: torch.Tensor,
    opacity_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    means, _, _ = _geometry(gaussians)
    feature_tensor = torch.as_tensor(features)
    if feature_tensor.ndim != 2 or feature_tensor.shape != (means.shape[0], 512):
        raise ValueError(
            f"features must have shape [{means.shape[0]},512], found {tuple(feature_tensor.shape)}"
        )
    if not feature_tensor.is_floating_point():
        raise ValueError("features must be floating point")
    expected_eligible = torch.as_tensor(
        eligibility_mask(
            gaussians, feature_tensor, opacity_threshold=opacity_threshold
        )
    ).detach().cpu()
    if expected_eligible.shape != (means.shape[0],) or expected_eligible.dtype != torch.bool:
        raise ValueError("calibration.eligibility_mask returned an invalid mask")

    metadata: dict[str, Any] = {}
    if isinstance(score_input, Mapping):
        if "scores" not in score_input or "eligible" not in score_input:
            raise ValueError("score bundle must contain 'scores' and 'eligible'")
        score_tensor = torch.as_tensor(score_input["scores"])
        supplied_eligible = torch.as_tensor(score_input["eligible"])
        if supplied_eligible.shape != expected_eligible.shape or supplied_eligible.dtype != torch.bool:
            raise ValueError("score bundle eligible mask has the wrong shape or dtype")
        supplied_eligible = supplied_eligible.detach().cpu()
        if not torch.equal(supplied_eligible, expected_eligible):
            raise ValueError(
                "score bundle eligibility differs from geometry/features/opacity_threshold"
            )
        raw_metadata = score_input.get("metadata", {})
        if not isinstance(raw_metadata, Mapping):
            raise ValueError("score bundle metadata must be a mapping")
        metadata = dict(raw_metadata)
        bundle_ids = _candidate_ids_from_metadata(score_input)
        metadata_ids = _candidate_ids_from_metadata(metadata)
        for mapped_ids in (bundle_ids, metadata_ids):
            if mapped_ids is not None and not torch.equal(mapped_ids, class_ids):
                raise ValueError(
                    "score candidate-column IDs do not exactly match the supplied class_ids order"
                )
        if "opacity_threshold" in metadata and not math.isclose(
            float(metadata["opacity_threshold"]),
            float(opacity_threshold),
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError(
                "score metadata opacity_threshold differs from evaluation opacity_threshold"
            )
    else:
        score_tensor = torch.as_tensor(score_input)

    if score_tensor.ndim != 2 or score_tensor.shape != (
        means.shape[0], class_ids.numel()
    ):
        raise ValueError(
            f"scores must have shape [{means.shape[0]},{class_ids.numel()}], "
            f"found {tuple(score_tensor.shape)}"
        )
    if not score_tensor.is_floating_point():
        raise ValueError("scores must be floating point")
    score_tensor = score_tensor.detach().cpu().to(dtype=torch.float32).contiguous()
    if expected_eligible.any() and not torch.isfinite(score_tensor[expected_eligible]).all():
        raise ValueError("eligible Gaussian scores must all be finite")
    if isinstance(score_input, Mapping) and (~expected_eligible).any():
        excluded = score_tensor[~expected_eligible]
        if not torch.isneginf(excluded).all():
            raise ValueError("calibration score bundles must use -inf for every excluded score")
    return score_tensor, expected_eligible.contiguous(), metadata


def _validate_assignments(
    assignments: Any,
    point_count: int,
    gaussian_count: int,
    eligible: torch.Tensor,
) -> torch.Tensor:
    result = _integer_vector(assignments, "assignments")
    if result.shape != (point_count,):
        raise ValueError(f"assignments must have shape [{point_count}], found {tuple(result.shape)}")
    if eligible.any():
        if (result < 0).any():
            raise ValueError("assignments may be -1 only when no Gaussian is eligible")
        if (result >= gaussian_count).any() or not eligible[result].all():
            raise ValueError("assignments contain an out-of-range or ineligible Gaussian")
    elif (result != -1).any():
        raise ValueError("all assignments must be -1 when no Gaussian is eligible")
    return result


def _class_row(class_id: int, tp: int, fp: int, fn: int, column: int) -> dict[str, Any]:
    support = tp + fn
    predicted = tp + fp
    union = tp + fp + fn
    return {
        "class_id": int(class_id),
        "column": int(column),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "support": int(support),
        "predicted": int(predicted),
        "iou": 100.0 * tp / union if union else 0.0,
        "precision": 100.0 * tp / predicted if predicted else 0.0,
        "recall": 100.0 * tp / support if support else 0.0,
    }


def _macro(per_class: Sequence[Mapping[str, Any]], key: str) -> float:
    if not per_class:
        return 0.0
    return math.fsum(float(row[key]) for row in per_class) / len(per_class)


def evaluate_scores(
    gaussians: Mapping[str, Any],
    features: Any,
    scores: Any,
    gt_points: Any,
    gt_labels: Any,
    class_ids: Any,
    *,
    mode: str = "multiclass",
    threshold: float | None = None,
    opacity_threshold: float = 0.1,
    assignments: Any | None = None,
    device: str | torch.device = "cuda",
) -> dict[str, Any]:
    """Evaluate fixed candidate scores on the full P3 ground-truth support.

    ``multiclass`` takes one argmax over every supplied candidate column and scores
    GT labels in that full candidate set. ``single_query`` thresholds each query
    independently and uses every nonignored GT label as binary support.
    """

    if mode not in {"multiclass", "single_query"}:
        raise ValueError("mode must be 'multiclass' or 'single_query'")
    if not math.isfinite(float(opacity_threshold)) or not 0.0 <= opacity_threshold <= 1.0:
        raise ValueError("opacity_threshold must be finite and in [0,1]")
    if mode == "single_query":
        if threshold is None or not math.isfinite(float(threshold)):
            raise ValueError("single_query evaluation requires a finite threshold")
        threshold = float(threshold)
    elif threshold is not None:
        raise ValueError("threshold is only defined for single_query evaluation")

    candidate_ids = _integer_vector(class_ids, "class_ids")
    if candidate_ids.numel() == 0:
        raise ValueError("class_ids must contain at least one fixed candidate")
    if torch.unique(candidate_ids).numel() != candidate_ids.numel():
        raise ValueError("class_ids must be unique; candidate columns cannot alias")
    if (candidate_ids < 0).any():
        raise ValueError("class_ids must be nonnegative dataset label IDs")

    points = torch.as_tensor(gt_points)
    if points.ndim != 2 or points.shape[1] != 3 or not points.is_floating_point():
        raise ValueError(f"gt_points must have shape [M,3], found {tuple(points.shape)}")
    if not torch.isfinite(points).all():
        raise ValueError("gt_points contain nonfinite coordinates")
    labels = _integer_vector(gt_labels, "gt_labels")
    if labels.shape != (points.shape[0],):
        raise ValueError("gt_points and gt_labels have different lengths")
    if (labels < -1).any():
        raise ValueError("gt_labels may use only -1 as the ignore label")
    means, _, _ = _geometry(gaussians)
    score_tensor, eligible, score_metadata = _prepare_scores(
        gaussians, features, scores, candidate_ids, float(opacity_threshold)
    )
    if "mode" in score_metadata and score_metadata["mode"] != mode:
        raise ValueError(
            f"score bundle mode {score_metadata['mode']!r} does not match evaluation mode {mode!r}"
        )

    assignment_source = "computed"
    if assignments is None:
        assignment_tensor = assign_mahalanobis(
            points, gaussians, eligible, device=device
        )
    else:
        assignment_tensor = _validate_assignments(
            assignments, points.shape[0], means.shape[0], eligible
        )
        assignment_source = "provided"
    assignment_tensor = _validate_assignments(
        assignment_tensor, points.shape[0], means.shape[0], eligible
    )

    candidate_population = torch.isin(labels, candidate_ids)
    nonignored_population = labels != -1
    present_columns = [
        column
        for column, class_id in enumerate(candidate_ids.tolist())
        if bool((labels == class_id).any())
    ]
    per_class: list[dict[str, Any]] = []
    owned = assignment_tensor >= 0

    if mode == "multiclass":
        predictions = torch.full_like(labels, -1)
        if owned.any():
            predicted_columns = score_tensor.index_select(
                0, assignment_tensor[owned]
            ).argmax(dim=1)
            predictions[owned] = candidate_ids.index_select(0, predicted_columns)
        for column in present_columns:
            class_id = int(candidate_ids[column])
            truth = (labels == class_id) & candidate_population
            predicted = (predictions == class_id) & candidate_population
            tp = int((truth & predicted).sum().item())
            fp = int((~truth & predicted & candidate_population).sum().item())
            fn = int((truth & ~predicted).sum().item())
            per_class.append(_class_row(class_id, tp, fp, fn, column))
        evaluated_population = candidate_population
    else:
        for column in present_columns:
            class_id = int(candidate_ids[column])
            positive_prediction = torch.zeros(labels.shape[0], dtype=torch.bool)
            if owned.any():
                point_scores = score_tensor[assignment_tensor[owned], column]
                positive_prediction[owned] = point_scores >= threshold
            truth = (labels == class_id) & nonignored_population
            predicted = positive_prediction & nonignored_population
            tp = int((truth & predicted).sum().item())
            fp = int((~truth & predicted & nonignored_population).sum().item())
            fn = int((truth & ~predicted).sum().item())
            per_class.append(_class_row(class_id, tp, fp, fn, column))
        evaluated_population = nonignored_population

    result = {
        "miou": _macro(per_class, "iou"),
        "precision": _macro(per_class, "precision"),
        "recall": _macro(per_class, "recall"),
        "per_class": per_class,
        "assignments": assignment_tensor,
        "provenance": {
            "schema": 1,
            "mode": mode,
            "threshold": threshold,
            "opacity_threshold": float(opacity_threshold),
            "assignment": {
                "algorithm": _ASSIGNMENT_ALGORITHM,
                "source": assignment_source,
                "tie_policy": "lowest original Gaussian index",
                "distance": "squared Mahalanobis; activated scales floored at 1e-8; float64-normalized WXYZ quaternion",
                "eligible_only": True,
                "unassigned_points": int((assignment_tensor < 0).sum().item()),
            },
            "candidate_class_ids": candidate_ids.tolist(),
            "candidate_columns": "scores[:, i] maps exactly to candidate_class_ids[i]",
            "present_class_ids": [
                int(candidate_ids[column]) for column in present_columns
            ],
            "macro_average": "original GT-present requested classes only",
            "population": {
                "total_gt_points": int(labels.numel()),
                "nonignored_gt_points": int(nonignored_population.sum().item()),
                "evaluated_gt_points": int(evaluated_population.sum().item()),
                "eligible_gaussians": int(eligible.sum().item()),
                "total_gaussians": int(eligible.numel()),
            },
            "score_metadata": score_metadata,
        },
    }
    return result


def _safe_mapping(path: str | Path, description: str) -> dict[str, Any]:
    value = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a dictionary: {path}")
    return value


def _validate_vocabulary(
    value: Mapping[str, Any], spec: DatasetVocabularySpec
) -> tuple[torch.Tensor, list[str], torch.Tensor, dict[str, list[int]]]:
    missing = {"class_ids", "names", "features", "subsets"} - set(value)
    if missing:
        raise ValueError(f"vocabulary is missing keys: {sorted(missing)}")
    class_ids = _integer_vector(value["class_ids"], "vocabulary class_ids")
    expected_ids = torch.tensor(spec.class_ids, dtype=torch.int64)
    if not torch.equal(class_ids, expected_ids):
        raise ValueError(
            f"{spec.dataset} vocabulary class_ids must be exactly {list(spec.class_ids)}"
        )
    names_value = value["names"]
    if not isinstance(names_value, (list, tuple)) or not all(
        isinstance(name, str) and name for name in names_value
    ):
        raise ValueError("vocabulary names must be nonempty strings")
    names = list(names_value)
    if len(names) != len(spec.class_ids) or len(set(names)) != len(names):
        raise ValueError("vocabulary names must be unique and aligned one-to-one with class_ids")
    if tuple(names) != spec.names:
        raise ValueError(
            f"{spec.dataset} vocabulary names differ from the fixed release order"
        )
    text_features = torch.as_tensor(value["features"])
    if text_features.dtype != torch.float32 or text_features.shape != (
        len(spec.class_ids),
        512,
    ):
        raise ValueError(
            f"vocabulary features must be float32[{len(spec.class_ids)},512]"
        )
    if not torch.isfinite(text_features).all():
        raise ValueError("vocabulary features contain nonfinite values")
    norms = torch.linalg.vector_norm(text_features, dim=1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-4, rtol=1e-4):
        raise ValueError("vocabulary features must be normalized")

    raw_subsets = value["subsets"]
    if not isinstance(raw_subsets, Mapping) or set(raw_subsets) != set(spec.subsets):
        raise ValueError(
            f"{spec.dataset} vocabulary subsets must be exactly {sorted(spec.subsets)}"
        )
    subsets: dict[str, list[int]] = {}
    for name, expected in spec.subsets.items():
        actual = _integer_vector(raw_subsets[name], f"vocabulary subset {name}")
        expected_tensor = torch.tensor(expected, dtype=torch.int64)
        if not torch.equal(actual, expected_tensor):
            raise ValueError(
                f"{spec.dataset} subset {name} must be exactly {list(expected)}"
            )
        subsets[name] = actual.tolist()
    return class_ids, names, text_features.contiguous(), subsets


def _selected_vocabulary(
    class_ids: torch.Tensor,
    names: list[str],
    subsets: Mapping[str, list[int]],
    subset: str,
) -> tuple[torch.Tensor, list[str]]:
    if subset == "full":
        return class_ids, names
    selected = torch.tensor(subsets[subset], dtype=torch.int64)
    positions = {int(class_id): index for index, class_id in enumerate(class_ids.tolist())}
    return selected, [names[positions[int(class_id)]] for class_id in selected]


