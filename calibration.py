#!/usr/bin/env python3
"""Standalone query-time calibration for canonical lifted CLIP feature fields.

The implementation follows the frozen calibration arithmetic: normalized scene-direction
subtraction, between-prototype covariance whitening, and a half-CSLS offset computed
independently for each score column.  Scene statistics and CSLS use every finite,
nonzero feature row, including low-opacity rows; opacity affects only output eligibility.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


FEATURE_DIM = 512
REFERENCE_SIZE = 45
REFERENCE_BANK_ID = "bank_35b9df6df6fb5bf5"
REFERENCE_WORDS = (
    "object", "stuff", "surface", "material", "background", "structure",
    "furniture", "room", "clutter", "corner", "space", "interior", "scene",
    "region", "pattern", "shape", "volume", "boundary", "arrangement",
    "composition", "portion", "section", "expanse", "form", "void", "span",
    "zone", "patch", "field", "bulk", "plane", "ridge", "groove", "seam",
    "face", "side", "depth", "extent", "scale", "spacing", "texture detail",
    "contour", "profile", "margin", "interval",
)
REFERENCE_INDICES = (
    0, 2, 4, 5, 6, 7, 8, 9, 11, 12, 14, 16, 17, 19, 20, 21, 22, 23, 25,
    26, 28, 29, 30, 31, 32, 34, 35, 37, 38, 40, 41, 42, 43, 44, 46, 47, 48,
    51, 52, 53, 55, 56, 58, 60, 61,
)
REFERENCE_SOURCE_SHA256 = "37092367ee6db13a24eb4a867cca5baffb8ef07cccb1708c2d4afa97a81f6ae5"
MODEL_NAME = "ViT-B-16"
MODEL_PRETRAINED = "laion2b_s34b_b88k"
NORMALIZE_EPS = 1e-12
EIGENVALUE_FLOOR_RELATIVE = 1e-10
HALF_CSLS = 0.5
_REFERENCE_PATH = Path(__file__).with_name("reference.pt")


@dataclass(frozen=True)
class SceneStatistics:
    """Reusable, CPU-resident scene inputs for scoring parameter grids.

    ``features`` are the canonical exported rows and are deliberately not normalized a
    second time.  This preserves the historical sub-epsilon rows produced by the original
    decoder's single ``F.normalize`` operation.
    """

    features: Tensor
    feature_valid: Tensor
    direction: Tensor
    valid_count: int


@dataclass(frozen=True)
class WhiteningStatistics:
    """Alpha-independent whitening fit reusable across an alpha grid."""

    mean: Tensor
    eigenvalues: Tensor
    eigenvectors: Tensor
    ridge_absolute: float
    eigenvalue_floor: float
    floor_count: int
    rows: int


def _require_tensor(value: Any, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    return value


def _as_cpu_float32(value: Tensor, name: str) -> Tensor:
    value = _require_tensor(value, name)
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    return value.detach().to(device="cpu", dtype=torch.float32).contiguous()


def load_gaussians(path: str | Path) -> dict[str, Tensor]:
    """Load and validate the standalone activated-Gaussian tensor dictionary."""

    source = Path(path)
    value = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(value, Mapping):
        raise TypeError(f"{source} must contain a tensor dictionary")
    required = {"means", "scales", "quats", "opacity"}
    missing = required.difference(value)
    extra = set(value).difference(required)
    if missing or extra:
        raise ValueError(
            f"{source} Gaussian keys differ: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    result = {name: _as_cpu_float32(value[name], name) for name in required}
    count = len(result["means"])
    expected = {
        "means": (count, 3),
        "scales": (count, 3),
        "quats": (count, 4),
        "opacity": (count,),
    }
    for name, shape in expected.items():
        if tuple(result[name].shape) != shape:
            raise ValueError(f"{source}:{name} has shape {tuple(result[name].shape)}, expected {shape}")
    return result


def load_features(path: str | Path) -> Tensor:
    """Load canonical CPU float32 feature rows without a second normalization."""

    source = Path(path)
    value = torch.load(source, map_location="cpu", weights_only=True)
    features = _as_cpu_float32(value, str(source))
    if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"{source} has shape {tuple(features.shape)}, expected [N,{FEATURE_DIM}]"
        )
    return features


def load_reference(path: str | Path | None = None) -> Tensor:
    """Safely load bundled G45 or an explicit normalized custom reference tensor."""

    bundled = path is None
    source = _REFERENCE_PATH if bundled else Path(path)
    reference = _as_cpu_float32(
        torch.load(source, map_location="cpu", weights_only=True), str(source)
    )
    if (
        reference.ndim != 2
        or not len(reference)
        or reference.shape[1] != FEATURE_DIM
        or (bundled and len(reference) != REFERENCE_SIZE)
    ):
        expected = (
            f"({REFERENCE_SIZE},{FEATURE_DIM})"
            if bundled
            else f"[rows,{FEATURE_DIM}] with rows > 0"
        )
        raise ValueError(f"{source} has shape {tuple(reference.shape)}, expected {expected}")
    if not bool(torch.isfinite(reference).all()):
        raise ValueError(f"{source} contains nonfinite reference values")
    norms = torch.linalg.vector_norm(reference, dim=1)
    if not bool(torch.allclose(norms, torch.ones_like(norms), rtol=0.0, atol=1e-5)):
        raise ValueError(f"{source} reference rows are not L2-normalized")
    return reference


def _feature_valid_mask(features: Tensor) -> Tensor:
    return torch.isfinite(features).all(dim=1) & (features != 0).any(dim=1)


def eligibility_mask(
    gaussians: Mapping[str, Tensor], features: Tensor, opacity_threshold: float = 0.1
) -> Tensor:
    """Return rows with finite geometry/features, nonzero features, and enough opacity.

    This mask is intentionally stricter than the statistics population: low-opacity rows
    are not eligible for output, but remain in the scene mean and CSLS population when
    their feature row is finite and nonzero.
    """

    if not math.isfinite(opacity_threshold):
        raise ValueError("opacity_threshold must be finite")
    required = ("means", "scales", "quats", "opacity")
    if any(name not in gaussians for name in required):
        raise ValueError(f"gaussians must contain {list(required)}")
    features = _as_cpu_float32(features, "features")
    count = len(features)
    geometry_finite = torch.ones(count, dtype=torch.bool)
    shapes = {"means": (count, 3), "scales": (count, 3), "quats": (count, 4), "opacity": (count,)}
    for name in required:
        tensor = _as_cpu_float32(gaussians[name], f"gaussians[{name!r}]")
        if tuple(tensor.shape) != shapes[name]:
            raise ValueError(
                f"gaussians[{name!r}] has shape {tuple(tensor.shape)}, expected {shapes[name]}"
            )
        geometry_finite &= torch.isfinite(tensor).reshape(count, -1).all(dim=1)
        if name == "opacity":
            opacity = tensor
    return geometry_finite & _feature_valid_mask(features) & (opacity >= opacity_threshold)


def compute_scene_statistics(features: Tensor, *, chunk_size: int = 16384) -> SceneStatistics:
    """Compute the canonical feature-valid population and scene direction once.

    The arithmetic is a chunked CPU float64 sum of the stored canonical rows, divided by
    row count, cast to float32, and L2-normalized.  It is independent of opacity, geometry,
    GT assignments, and requested queries.
    """

    features = _as_cpu_float32(features, "features")
    if features.ndim != 2 or not features.shape[1]:
        raise ValueError("features must have shape [N,D] with D > 0")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    valid = _feature_valid_mask(features)
    count = int(valid.sum().item())
    if not count:
        return SceneStatistics(
            features=features,
            feature_valid=valid,
            direction=torch.zeros((1, features.shape[1]), dtype=torch.float32),
            valid_count=0,
        )
    total = torch.zeros((1, features.shape[1]), dtype=torch.float64)
    seen = 0
    for start in range(0, len(features), chunk_size):
        stop = min(start + chunk_size, len(features))
        local = valid[start:stop]
        if bool(local.any()):
            rows = features[start:stop][local]
            total += rows.sum(dim=0, keepdim=True, dtype=torch.float64)
            seen += len(rows)
    if seen != count:
        raise RuntimeError(f"scene reduction saw {seen} rows, expected {count}")
    direction = F.normalize((total / count).float(), dim=-1, eps=NORMALIZE_EPS)
    if not bool(torch.isfinite(direction).all()):
        raise ValueError("scene direction is nonfinite")
    return SceneStatistics(features=features, feature_valid=valid, direction=direction, valid_count=count)


def center_rows(rows: Tensor, scene_direction: Tensor, lambda_: float) -> Tensor:
    """Apply canonical normalized scene-direction subtraction to row vectors."""

    if not math.isfinite(lambda_):
        raise ValueError("lambda_ must be finite")
    if rows.ndim != 2 or scene_direction.shape != (1, rows.shape[1]):
        raise ValueError("rows and scene_direction shapes disagree")
    shifted = rows - float(lambda_) * scene_direction.to(device=rows.device, dtype=rows.dtype)
    centered = F.normalize(shifted, dim=-1, eps=NORMALIZE_EPS)
    if not bool(torch.isfinite(centered).all()):
        raise ValueError("scene-centered rows are nonfinite")
    return centered


def fit_whitening(rows: Tensor, *, ridge: float = 0.001) -> WhiteningStatistics:
    """Fit canonical between-prototype covariance statistics, independent of alpha.

    Covariance and ridge are float32, the symmetric eigendecomposition is float64, and
    eigenvalues are floored relative to the largest regularized eigenvalue.
    """

    if rows.ndim != 2 or not len(rows) or not rows.is_floating_point():
        raise ValueError("rows must be a nonempty rank-2 floating-point tensor")
    if not math.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and nonnegative")
    if not bool(torch.isfinite(rows).all()):
        raise ValueError("whitening fit rows contain nonfinite values")
    mean = rows.mean(dim=0, keepdim=True)
    centered = rows - mean
    covariance = (centered.T @ centered) / max(len(rows) - 1, 1)
    ridge_tensor = float(ridge) * torch.trace(covariance) / covariance.shape[0]
    regularized = covariance + ridge_tensor * torch.eye(
        covariance.shape[0], device=rows.device, dtype=rows.dtype
    )
    eigenvalues0, eigenvectors = torch.linalg.eigh(regularized.double())
    maximum = float(eigenvalues0.max().item())
    if not math.isfinite(maximum) or maximum <= 0:
        raise ValueError(
            "whitening fit has zero covariance; use at least two nonidentical fit rows "
            "or disable whitening"
        )
    floor = EIGENVALUE_FLOOR_RELATIVE * maximum
    eigenvalues = eigenvalues0.clamp_min(floor)
    return WhiteningStatistics(
        mean=mean,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        ridge_absolute=float(ridge_tensor.item()),
        eigenvalue_floor=floor,
        floor_count=int((eigenvalues0 < eigenvalues).sum().item()),
        rows=len(rows),
    )


def whitening_matrix(statistics: WhiteningStatistics, alpha: float) -> Tensor:
    """Construct the canonical dense whitening matrix for one alpha value."""

    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    values = statistics.eigenvalues ** (-float(alpha))
    matrix = (
        statistics.eigenvectors
        @ torch.diag(values)
        @ statistics.eigenvectors.T
    ).float()
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError("whitening matrix is nonfinite")
    return matrix


def _apply_whitening_matrix(
    rows: Tensor, statistics: WhiteningStatistics, matrix: Tensor
) -> Tensor:
    if rows.ndim != 2 or rows.shape[1] != statistics.mean.shape[1]:
        raise ValueError("rows and whitening statistics shapes disagree")
    transformed = F.normalize(
        (rows - statistics.mean) @ matrix.T, dim=-1, eps=NORMALIZE_EPS
    )
    if not bool(torch.isfinite(transformed).all()):
        raise ValueError("whitened rows are nonfinite")
    return transformed


def apply_whitening(rows: Tensor, statistics: WhiteningStatistics, alpha: float) -> Tensor:
    """Apply a reusable whitening fit and L2-normalize the transformed rows."""

    return _apply_whitening_matrix(
        rows, statistics, whitening_matrix(statistics, alpha)
    )


def encode_text_queries(
    queries: Sequence[str], *, device: str | torch.device = "cuda", batch_size: int = 64
) -> Tensor:
    """Encode bare phrases with canonical OpenCLIP settings.

    ``open_clip`` is imported only when text encoding is requested.  Tensor-query users do
    not need that optional dependency or its model weights.
    """

    names = list(queries)
    if not names or any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("text queries must be a nonempty sequence of nonempty strings")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    try:
        import open_clip  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "text queries require the optional open_clip_torch package; tensor queries do not"
        ) from error
    device = torch.device(device)
    model, _, _ = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=MODEL_PRETRAINED, device=device
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    encoded: list[Tensor] = []
    context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else nullcontext()
    )
    phrases = [name.replace("shower curtain", "showercurtain") for name in names]
    with torch.inference_mode(), context:
        for start in range(0, len(phrases), batch_size):
            tokens = tokenizer(phrases[start:start + batch_size]).to(device)
            rows = model.encode_text(tokens).float()
            encoded.append(F.normalize(rows, dim=-1, eps=NORMALIZE_EPS).cpu())
    result = torch.cat(encoded).contiguous()
    if result.shape != (len(names), FEATURE_DIM) or not bool(torch.isfinite(result).all()):
        raise ValueError("OpenCLIP returned invalid query embeddings")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _normalize_query_rows(rows: Tensor, dimension: int) -> Tensor:
    rows = _as_cpu_float32(rows, "queries")
    if rows.ndim != 2 or not len(rows) or rows.shape[1] != dimension:
        raise ValueError(f"queries must have shape [C,{dimension}] with C > 0")
    valid = torch.isfinite(rows).all(dim=1) & (rows != 0).any(dim=1)
    if not bool(valid.all()):
        raise ValueError("query rows must be finite and exactly nonzero")
    norms = torch.linalg.vector_norm(rows, dim=1)
    if bool(torch.allclose(norms, torch.ones_like(norms), rtol=0.0, atol=1e-5)):
        return rows
    return F.normalize(rows, dim=-1, eps=NORMALIZE_EPS).contiguous()


def _validate_normalized_rows(rows: Tensor, dimension: int, name: str) -> Tensor:
    rows = _as_cpu_float32(rows, name)
    if rows.ndim != 2 or not len(rows) or rows.shape[1] != dimension:
        raise ValueError(f"{name} must have shape [rows,{dimension}] with rows > 0")
    valid = torch.isfinite(rows).all(dim=1) & (rows != 0).any(dim=1)
    if not bool(valid.all()):
        raise ValueError(f"{name} rows must be finite and exactly nonzero")
    norms = torch.linalg.vector_norm(rows, dim=1)
    if not bool(torch.allclose(norms, torch.ones_like(norms), rtol=0.0, atol=1e-5)):
        raise ValueError(f"{name} rows must already be L2-normalized")
    return rows




def _whitening_metadata(statistics: WhiteningStatistics) -> dict[str, Any]:
    return {
        "fit_rows": statistics.rows,
        "ridge_absolute": statistics.ridge_absolute,
        "eigenvalue_floor_relative": EIGENVALUE_FLOOR_RELATIVE,
        "eigenvalue_floor_absolute": statistics.eigenvalue_floor,
        "floored_eigenvalues": statistics.floor_count,
    }


def _prepare_prototypes(
    queries: Tensor,
    reference: Tensor,
    direction: Tensor,
    *,
    lambda_: float,
    alpha: float,
    ridge: float,
    center: bool,
    whiten: bool,
    fit: str,
    mode: str,
    device: torch.device,
) -> tuple[Tensor, Tensor | None, Tensor | None, dict[str, Any]]:
    """Return transformed queries, shared references, or independent N1 groups."""

    query = queries.to(device)
    bank = reference.to(device)
    mu = direction.to(device)
    centered_query = center_rows(query, mu, lambda_) if center else query
    centered_bank = center_rows(bank, mu, lambda_) if center and len(bank) else bank
    diagnostics: dict[str, Any] = {
        "fit": fit,
        "fit_requested_rows": None,
        "fit_reference_rows": None,
        "independent_fits": False,
    }
    if not whiten:
        diagnostics.update({"operator": "disabled", "fit_requested_rows": 0, "fit_reference_rows": 0})
        return centered_query, centered_bank if mode == "single_query" else None, None, diagnostics

    if fit == "reference":
        if not len(centered_bank):
            raise ValueError("fit='reference' requires at least two reference rows")
        operator = fit_whitening(centered_bank, ridge=ridge)
        matrix = whitening_matrix(operator, alpha)
        transformed_query = _apply_whitening_matrix(
            centered_query, operator, matrix
        )
        transformed_bank = _apply_whitening_matrix(
            centered_bank, operator, matrix
        )
        diagnostics.update({
            "fit_requested_rows": 0,
            "fit_reference_rows": len(centered_bank),
            "operator": _whitening_metadata(operator),
        })
        return (
            transformed_query,
            transformed_bank if mode == "single_query" else None,
            None,
            diagnostics,
        )

    if fit == "query":
        if mode != "multiclass":
            raise ValueError("fit='query' is the bank-removed multiclass control only")
        operator = fit_whitening(centered_query, ridge=ridge)
        transformed_query = apply_whitening(centered_query, operator, alpha)
        diagnostics.update({
            "fit_requested_rows": len(centered_query),
            "fit_reference_rows": 0,
            "operator": _whitening_metadata(operator),
        })
        return transformed_query, None, None, diagnostics

    if fit != "query_reference":
        raise ValueError("fit must be 'reference', 'query_reference', or 'query'")

    if mode == "multiclass":
        combined = torch.cat((centered_query, centered_bank), dim=0)
        operator = fit_whitening(combined, ridge=ridge)
        transformed = apply_whitening(combined, operator, alpha)
        diagnostics.update({
            "fit_requested_rows": len(centered_query),
            "fit_reference_rows": len(centered_bank),
            "operator": _whitening_metadata(operator),
        })
        return transformed[:len(centered_query)], None, None, diagnostics

    if not len(centered_bank):
        raise ValueError("single_query query_reference fitting requires a reference bank")
    groups: list[Tensor] = []
    operator_summaries = []
    for query_row in centered_query:
        combined = torch.cat((query_row[None], centered_bank), dim=0)
        operator = fit_whitening(combined, ridge=ridge)
        groups.append(apply_whitening(combined, operator, alpha))
        operator_summaries.append(_whitening_metadata(operator))
    independent = torch.stack(groups)
    diagnostics.update({
        "fit_requested_rows": 1,
        "fit_reference_rows": len(centered_bank),
        "independent_fits": True,
        "independent_fit_count": len(centered_query),
        "operator": {
            "fit_rows": 1 + len(centered_bank),
            "ridge_absolute_min": min(row["ridge_absolute"] for row in operator_summaries),
            "ridge_absolute_max": max(row["ridge_absolute"] for row in operator_summaries),
            "eigenvalue_floor_relative": EIGENVALUE_FLOOR_RELATIVE,
            "floored_eigenvalues_min": min(row["floored_eigenvalues"] for row in operator_summaries),
            "floored_eigenvalues_max": max(row["floored_eigenvalues"] for row in operator_summaries),
        },
    })
    return independent[:, 0], None, independent, diagnostics


def _visual_rows(
    rows: Tensor, direction: Tensor, lambda_: float, center: bool, device: torch.device
) -> Tensor:
    rows = rows.to(device)
    return center_rows(rows, direction.to(device), lambda_) if center else rows


def _column_offsets(
    scene: SceneStatistics,
    prototypes: Tensor,
    *,
    direction: Tensor,
    lambda_: float,
    center: bool,
    csls_k: int,
    device: torch.device,
    chunk_size: int,
) -> Tensor:
    k = min(csls_k, scene.valid_count)
    if k <= 0:
        raise ValueError("csls_k and the feature-valid population must be positive")
    matrix = prototypes.T.contiguous()
    top = torch.empty((0, len(prototypes)), dtype=torch.float32, device=device)
    for start in range(0, len(scene.features), chunk_size):
        stop = min(start + chunk_size, len(scene.features))
        local = scene.feature_valid[start:stop]
        if not bool(local.any()):
            continue
        visual = _visual_rows(
            scene.features[start:stop][local], direction, lambda_, center, device
        )
        scores = visual @ matrix
        candidates = scores.topk(min(k, len(scores)), dim=0).values
        top = torch.cat((top, candidates), dim=0).topk(
            min(k, len(top) + len(candidates)), dim=0
        ).values
    if len(top) != k or not bool(torch.isfinite(top).all()):
        raise RuntimeError(f"CSLS retained {len(top)} rows, expected {k}")
    return HALF_CSLS * top.mean(dim=0)


def _score_matrix(
    scene: SceneStatistics,
    eligible: Tensor,
    prototypes: Tensor,
    *,
    direction: Tensor,
    lambda_: float,
    center: bool,
    csls: bool,
    csls_k: int,
    device: torch.device,
    chunk_size: int,
) -> tuple[Tensor, dict[str, Any]]:
    offsets = (
        _column_offsets(
            scene, prototypes, direction=direction, lambda_=lambda_, center=center,
            csls_k=csls_k, device=device, chunk_size=chunk_size,
        )
        if csls
        else torch.zeros(len(prototypes), dtype=torch.float32, device=device)
    )
    matrix = prototypes.T.contiguous()
    result = torch.full((len(scene.features), len(prototypes)), -torch.inf, dtype=torch.float32)
    for start in range(0, len(scene.features), chunk_size):
        stop = min(start + chunk_size, len(scene.features))
        local = eligible[start:stop]
        if not bool(local.any()):
            continue
        visual = _visual_rows(
            scene.features[start:stop][local], direction, lambda_, center, device
        )
        values = visual @ matrix - offsets
        result[start:stop][local] = values.cpu()
    return result, {
        "csls_k_effective": min(csls_k, scene.valid_count) if csls else None,
        "offset_min": float(offsets.min().item()) if len(offsets) else None,
        "offset_max": float(offsets.max().item()) if len(offsets) else None,
        "score_columns": len(prototypes),
    }


def _score_shared_margin(
    scene: SceneStatistics,
    eligible: Tensor,
    queries: Tensor,
    reference: Tensor,
    **kwargs: Any,
) -> tuple[Tensor, dict[str, Any]]:
    combined = torch.cat((queries, reference), dim=0)
    scores, diagnostics = _score_matrix(scene, eligible, combined, **kwargs)
    query_count = len(queries)
    margins = scores[:, :query_count] - scores[:, query_count:].max(dim=1, keepdim=True).values
    margins[~eligible] = -torch.inf
    diagnostics.update({
        "score_columns": len(combined),
        "output_columns": query_count,
        "reference_score_columns": len(reference),
        "margin": "target minus maximum reference after per-column CSLS",
    })
    return margins, diagnostics


def _score_independent_margins(
    scene: SceneStatistics,
    eligible: Tensor,
    groups: Tensor,
    *,
    direction: Tensor,
    lambda_: float,
    center: bool,
    csls: bool,
    csls_k: int,
    device: torch.device,
    chunk_size: int,
) -> tuple[Tensor, dict[str, Any]]:
    query_count, closed_size, dimension = groups.shape
    if closed_size < 2 or dimension != scene.features.shape[1]:
        raise ValueError("invalid independent single-query prototype groups")
    result = torch.full((len(scene.features), query_count), -torch.inf, dtype=torch.float32)
    max_score_elements = 16 * 1024 * 1024
    queries_per_block = max(1, max_score_elements // max(chunk_size * closed_size, 1))
    queries_per_block = min(query_count, queries_per_block)
    offset_min = None
    offset_max = None
    for first in range(0, query_count, queries_per_block):
        last = min(first + queries_per_block, query_count)
        prototypes = groups[first:last].reshape(-1, dimension)
        offsets = (
            _column_offsets(
                scene, prototypes, direction=direction, lambda_=lambda_, center=center,
                csls_k=csls_k, device=device, chunk_size=chunk_size,
            )
            if csls
            else torch.zeros(len(prototypes), dtype=torch.float32, device=device)
        )
        offset_min = float(offsets.min().item()) if offset_min is None else min(offset_min, float(offsets.min().item()))
        offset_max = float(offsets.max().item()) if offset_max is None else max(offset_max, float(offsets.max().item()))
        matrix = prototypes.T.contiguous()
        width = last - first
        for start in range(0, len(scene.features), chunk_size):
            stop = min(start + chunk_size, len(scene.features))
            local = eligible[start:stop]
            if not bool(local.any()):
                continue
            visual = _visual_rows(
                scene.features[start:stop][local], direction, lambda_, center, device
            )
            scores = (visual @ matrix - offsets).reshape(-1, width, closed_size)
            margins = scores[:, :, 0] - scores[:, :, 1:].max(dim=2).values
            result[start:stop][local, first:last] = margins.cpu()
    return result, {
        "csls_k_effective": min(csls_k, scene.valid_count) if csls else None,
        "offset_min": offset_min,
        "offset_max": offset_max,
        "independent_closed_vocabulary": [query_count, closed_size, dimension],
        "queries_per_score_block": queries_per_block,
        "margin": "each independently fitted target minus its own maximum reference",
    }


def score_embeddings(
    scene: SceneStatistics,
    eligible: Tensor,
    queries: Tensor,
    *,
    reference: Tensor | None = None,
    lambda_: float = 0.3,
    alpha: float = 0.4,
    ridge: float = 0.001,
    csls_k: int = 1000,
    mode: str = "multiclass",
    raw: bool = False,
    center: bool = True,
    whiten: bool = True,
    csls: bool = True,
    fit: str = "reference",
    device: str | torch.device = "cuda",
    chunk_size: int = 16384,
) -> tuple[Tensor, dict[str, Any]]:
    """Score already-encoded queries against reusable scene statistics."""

    if mode not in {"multiclass", "single_query"}:
        raise ValueError("mode must be 'multiclass' or 'single_query'")
    if fit not in {"reference", "query_reference", "query"}:
        raise ValueError("fit must be 'reference', 'query_reference', or 'query'")
    if not raw and fit == "query" and mode != "multiclass":
        raise ValueError("fit='query' is the bank-removed multiclass control only")
    if chunk_size <= 0 or csls_k <= 0:
        raise ValueError("chunk_size and csls_k must be positive")
    if eligible.dtype != torch.bool or eligible.device.type != "cpu" or eligible.shape != scene.feature_valid.shape:
        raise ValueError("eligible must be a CPU bool tensor aligned to scene features")
    if bool((eligible & ~scene.feature_valid).any()):
        raise ValueError("eligible contains a feature-invalid row")
    queries = _validate_normalized_rows(
        queries, scene.features.shape[1], "queries"
    )
    if not scene.valid_count:
        return (
            torch.full(
                (len(scene.features), len(queries)), -torch.inf, dtype=torch.float32
            ),
            {
                "prototype_transform": {
                    "operator": "not_run",
                    "reason": "no_finite_nonzero_feature_rows",
                },
                "scoring": {
                    "status": "no_valid_features",
                    "csls_k_effective": 0,
                    "score_columns": len(queries),
                },
            },
        )
    reference_rows = (
        load_reference()
        if reference is None
        else _validate_normalized_rows(
            reference, scene.features.shape[1], "reference"
        )
    )
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    if raw:
        raw_scores, scoring = _score_matrix(
            scene, eligible, queries.to(device), direction=scene.direction,
            lambda_=lambda_, center=False, csls=False, csls_k=csls_k,
            device=device, chunk_size=chunk_size,
        )
        return raw_scores, {
            "prototype_transform": {"operator": "raw cosine", "fit": "none"},
            "scoring": scoring,
        }

    transformed_queries, transformed_reference, independent, prototype_metadata = _prepare_prototypes(
        queries, reference_rows, scene.direction, lambda_=lambda_, alpha=alpha,
        ridge=ridge, center=center, whiten=whiten, fit=fit, mode=mode, device=device,
    )
    common = {
        "direction": scene.direction,
        "lambda_": lambda_,
        "center": center,
        "csls": csls,
        "csls_k": csls_k,
        "device": device,
        "chunk_size": chunk_size,
    }
    if mode == "multiclass":
        scores, scoring = _score_matrix(
            scene, eligible, transformed_queries, **common
        )
    elif independent is not None:
        scores, scoring = _score_independent_margins(
            scene, eligible, independent, **common
        )
    else:
        if transformed_reference is None or not len(transformed_reference):
            raise ValueError("single_query scoring requires reference score columns")
        scores, scoring = _score_shared_margin(
            scene, eligible, transformed_queries, transformed_reference, **common
        )
    return scores, {"prototype_transform": prototype_metadata, "scoring": scoring}


def _tensor_sha256(value: Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Tensor):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def score_queries(
    gaussians: Mapping[str, Tensor],
    features: Tensor,
    queries: Tensor | Sequence[str],
    *,
    reference: Tensor | None = None,
    lambda_: float = 0.3,
    alpha: float = 0.4,
    ridge: float = 0.001,
    csls_k: int = 1000,
    mode: str = "multiclass",
    raw: bool = False,
    center: bool = True,
    whiten: bool = True,
    csls: bool = True,
    fit: str = "reference",
    opacity_threshold: float = 0.1,
    device: str | torch.device = "cuda",
    chunk_size: int = 16384,
) -> dict[str, Any]:
    """Calibrate query scores and return a safe, CPU-resident score bundle.

    In ``single_query`` mode, reference-only fitting applies one fixed reference
    operator to every query and returns target-minus-maximum-reference margins.
    With ``fit='query_reference'``, each query is instead fitted independently
    with the same ordered reference rows; requested queries never enter one
    another's fit. Multiclass joint fitting uses all requested candidates and
    the reference bank in one fit, then drops reference rows before score
    columns and CSLS are computed. ``fit='query'`` is the bank-removed
    multiclass control.
    """

    features = _as_cpu_float32(features, "features")
    if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
        raise ValueError(f"features must have shape [N,{FEATURE_DIM}]")
    if mode not in {"multiclass", "single_query"}:
        raise ValueError("mode must be 'multiclass' or 'single_query'")
    if fit not in {"reference", "query_reference", "query"}:
        raise ValueError("fit must be 'reference', 'query_reference', or 'query'")
    if not raw and fit == "query" and mode != "multiclass":
        raise ValueError("fit='query' is the bank-removed multiclass control only")
    if isinstance(queries, Tensor):
        encoded_queries = _normalize_query_rows(queries, features.shape[1])
        query_names = None
        query_source = "tensor"
    else:
        if isinstance(queries, str):
            query_names = [queries]
        else:
            query_names = list(queries)
        encoded_queries = encode_text_queries(query_names, device=device)
        query_source = "canonical_openclip_bare_phrase"
    canonical_reference = load_reference()
    reference_rows = (
        canonical_reference
        if reference is None
        else _validate_normalized_rows(reference, features.shape[1], "reference")
    )
    reference_is_canonical = (
        reference_rows.shape == canonical_reference.shape
        and torch.equal(reference_rows, canonical_reference)
    )
    scene = compute_scene_statistics(features, chunk_size=chunk_size)
    eligible = eligibility_mask(gaussians, features, opacity_threshold=opacity_threshold)
    if scene.valid_count:
        scores, diagnostics = score_embeddings(
            scene, eligible, encoded_queries, reference=reference_rows, lambda_=lambda_,
            alpha=alpha, ridge=ridge, csls_k=csls_k, mode=mode, raw=raw,
            center=center, whiten=whiten, csls=csls, fit=fit, device=device,
            chunk_size=chunk_size,
        )
    else:
        scores = torch.full(
            (len(features), len(encoded_queries)), -torch.inf, dtype=torch.float32
        )
        eligible = torch.zeros(len(features), dtype=torch.bool)
        diagnostics = {
            "prototype_transform": {
                "operator": "not_run",
                "reason": "no_finite_nonzero_feature_rows",
            },
            "scoring": {
                "status": "no_valid_features",
                "csls_k_effective": 0,
                "score_columns": len(encoded_queries),
            },
        }
    metadata = {
        "schema": 1,
        "mode": mode,
        "query_count": len(encoded_queries),
        "query_names": query_names,
        "query_source": query_source,
        "feature_dimension": features.shape[1],
        "parameters": {
            "lambda": float(lambda_), "alpha": float(alpha), "ridge": float(ridge),
            "csls_k": int(csls_k), "opacity_threshold": float(opacity_threshold),
            "raw": bool(raw), "center": bool(center and not raw),
            "whiten": bool(whiten and not raw), "csls": bool(csls and not raw),
            "fit": "none" if raw else fit, "chunk_size": int(chunk_size),
        },
        "population": {
            "primitives": len(features),
            "feature_valid": scene.valid_count,
            "eligible": int(eligible.sum().item()),
            "status": "ready" if scene.valid_count else "no_valid_features",
            "calibration_executed": bool(scene.valid_count),
            "statistics": "all finite exact-nonzero feature rows, including low-opacity rows",
            "output": "finite score only for finite-geometry, feature-valid, opacity-eligible rows; -inf otherwise",
        },
        "normalization": {
            "features": "preserved canonical exported rows; no second L2 normalization",
            "feature_zero_test": "exactly all-zero row after finite check",
            "queries": "canonical unit rows preserved; other finite nonzero tensor rows float32 L2-normalized once at API boundary",
            "scene_direction": (
                "CPU float64 sum/mean of stored feature-valid rows, float32 L2 normalization"
                if scene.valid_count
                else "zero-vector sentinel; no whitening, CSLS, or scoring executed"
            ),
            "centered_rows": "L2-normalize(row - lambda * scene_direction), eps=1e-12",
            "whitening": "float32 between-prototype covariance; trace-scaled ridge; float64 eigh; relative 1e-10 floor; float32 dense projection; L2 normalization",
        },
        "reference": {
            "kind": "canonical_g45" if reference_is_canonical else "custom",
            "canonical_exact_match": reference_is_canonical,
            "bank_id": REFERENCE_BANK_ID if reference_is_canonical else None,
            "words": list(REFERENCE_WORDS) if reference_is_canonical else None,
            "canonical_indices": (
                list(REFERENCE_INDICES) if reference_is_canonical else None
            ),
            "rows": len(reference_rows),
            "tensor_sha256": _tensor_sha256(reference_rows),
            "source_npz_sha256": (
                REFERENCE_SOURCE_SHA256 if reference_is_canonical else None
            ),
        },
        "encoder": {
            "model": MODEL_NAME, "pretrained": MODEL_PRETRAINED,
            "prompt": "bare phrase", "autocast": "float16 on CUDA",
            "override": {"shower curtain": "showercurtain"},
        } if query_names is not None else None,
        **diagnostics,
    }
    return {
        "scores": scores.cpu().float(),
        "eligible": eligible.cpu(),
        "metadata": _json_safe(metadata),
    }


def _load_cli_queries(path: Path) -> tuple[Tensor, dict[str, Any]]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    metadata: dict[str, Any] = {"query_path": str(path.resolve())}
    if isinstance(value, Tensor):
        return value, metadata
    if isinstance(value, Mapping) and isinstance(value.get("features"), Tensor):
        for key in ("names", "class_ids", "subsets"):
            if key in value:
                metadata[key] = _json_safe(value[key])
        return value["features"], metadata
    raise TypeError(f"{path} must contain a tensor or a dictionary with tensor key 'features'")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gaussians", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--queries", type=Path, help="safe .pt tensor or vocabulary dictionary")
    source.add_argument("--text", action="append", help="bare text query; repeat for multiple queries")
    parser.add_argument("--reference", type=Path, help="optional safe .pt reference tensor")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("multiclass", "single_query"), default="multiclass")
    parser.add_argument("--fit", choices=("reference", "query_reference", "query"), default="reference")
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--ridge", type=float, default=0.001)
    parser.add_argument("--csls-k", type=int, default=1000)
    parser.add_argument("--opacity-threshold", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-size", type=int, default=16384)
    parser.add_argument("--raw", action="store_true", help="raw cosine; overrides all calibration components")
    parser.add_argument("--center", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--whiten", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--csls", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    gaussians = load_gaussians(args.gaussians)
    features = load_features(args.features)
    cli_metadata: dict[str, Any]
    if args.queries is not None:
        queries, cli_metadata = _load_cli_queries(args.queries)
    else:
        queries = list(args.text)
        cli_metadata = {"query_text": queries}
    reference = load_reference(args.reference) if args.reference else None
    result = score_queries(
        gaussians, features, queries, reference=reference, lambda_=args.lambda_,
        alpha=args.alpha, ridge=args.ridge, csls_k=args.csls_k, mode=args.mode,
        raw=args.raw, center=args.center, whiten=args.whiten, csls=args.csls,
        fit=args.fit, opacity_threshold=args.opacity_threshold, device=args.device,
        chunk_size=args.chunk_size,
    )
    result["metadata"]["cli"] = {
        "gaussians": str(args.gaussians.resolve()),
        "features": str(args.features.resolve()),
        **cli_metadata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "scores": list(result["scores"].shape),
        "eligible": int(result["eligible"].sum().item()),
        "mode": args.mode,
        "fit": "none" if args.raw else args.fit,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
