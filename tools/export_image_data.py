#!/usr/bin/env python3
"""Export self-contained image benchmarks from explicitly supplied offline sources.

Conversion only: --source-root must contain the archived evaluation_workspaces
sources (or be evaluation_workspaces itself), and --asset-root must contain the
user-provided lerf_ovs, lerf_ovs_raw, 3dovs_baselines, 3dovs_recons and 3dovs_raw
bundles. Neither dependency is needed by the canonical evaluator. No training,
source checkpoint duplication, SH export, or evaluator execution is performed.
Existing point datasets and their manifest entries are preserved.
Resume requires matching Torch/CUDA/math generation identity and source/output
hashes. Runtime refresh regenerates features from original sources (including
LangSplat decoding), never renormalizes canonical features, and retains shared
assets and geometry/validity files only when their contents are unchanged.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

METHODS = ("sfs", "normlift", "ludvig", "occam", "vala", "opengaussian", "langsplat")
SCENES = {
    "lerf_ovs": ("figurines", "ramen", "teatime", "waldo_kitchen"),
    "3d_ovs": ("bed", "bench", "blue_sofa", "covered_desk", "lawn", "office_desk", "room", "snacks", "sofa", "table"),
}
WORKSPACES = {"lerf_ovs": "lerf_langsplat_20260925", "3d_ovs": "3d_ovs_langsplat_20260925"}


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def save_tensor(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


class Hashes:
    def __init__(self) -> None:
        self.cache: dict[tuple[str, int, int], dict[str, Any]] = {}

    def file(self, path: Path) -> dict[str, Any]:
        stat = path.stat()
        key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
        if key not in self.cache:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            self.cache[key] = {"sha256": digest.hexdigest(), "size": stat.st_size}
        return dict(self.cache[key])


def array_hash(tensor: torch.Tensor) -> str:
    array = tensor.contiguous().numpy()
    digest = hashlib.sha256(str((array.shape, array.dtype.str)).encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def modules(source: Path, dataset: str) -> dict[str, Any]:
    folder = source / WORKSPACES[dataset] / "source"
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    # Source imports are an explicit offline conversion dependency, never copied
    # into or imported by the runtime evaluator.
    sys.path.insert(0, str(folder))
    if dataset == "lerf_ovs":
        names = ("lerf_gaussian", "lerf_normvala", "lerf_opengaussian", "lerf_langsplat", "lerf_scoring")
    else:
        names = ("ovs_gaussian", "ovs_data", "ovs_scoring")
    return {name: importlib.import_module(name) for name in names}


def vocabulary(cache: Path, queries: list[str], dataset: str) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = torch.load(cache, map_location="cpu", weights_only=True)
    metadata = payload["metadata"]
    names = metadata["queries"]
    rows = payload["features" if dataset == "lerf_ovs" else "text_features"]
    expected = metadata["feature_sha256" if dataset == "lerf_ovs" else "text_feature_sha256"]
    if rows.dtype != torch.float32 or rows.shape != (len(names), 512) or array_hash(rows) != expected:
        raise ValueError(f"invalid frozen vocabulary cache: {cache}")
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate frozen vocabulary entries: {cache}")
    if dataset == "lerf_ovs":
        encoder = {"model": metadata["model"], "pretrained": metadata["pretrained"]}
    else:
        encoder = metadata["encoder"]
    if encoder["model"] != "ViT-B-16" or encoder["pretrained"] != "laion2b_s34b_b88k":
        raise ValueError(f"wrong frozen text encoder: {cache}")
    lookup = {name: index for index, name in enumerate(names)}
    selected = rows[[lookup[name] for name in queries]].contiguous()
    if not torch.isfinite(selected).all() or not torch.allclose(selected.norm(dim=1), torch.ones(len(queries)), atol=1e-5, rtol=0):
        raise ValueError(f"non-unit frozen text rows: {cache}")
    return {"names": queries, "features": selected, "class_ids": torch.arange(len(queries), dtype=torch.int64)}, metadata


def numerical_runtime(device: str, chunk: int) -> dict[str, Any]:
    return {"torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
            "numpy": str(np.__version__), "device": device, "chunk_size": chunk,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32)}


def reusable_shared_scene(dataset: str, scene: str, asset: Path, source: Path, target: Path, hashes: Hashes) -> dict[str, Any] | None:
    directory = target / dataset / scene
    scene_path, metadata_path = directory / "scene.json", directory / "metadata.json"
    manifest_path = target / dataset / "manifest.json"
    if not all(path.is_file() for path in (scene_path, metadata_path, manifest_path)):
        return None
    metadata = json.loads(metadata_path.read_text())
    sources = metadata.get("source_files", {})
    cache = (source / WORKSPACES[dataset] / "text_cache.pt").resolve()
    if not sources or not any(Path(path).resolve() == cache for path in sources):
        return None
    for name, info in sources.items():
        path = Path(name)
        if not (path.resolve() == cache or path.resolve().is_relative_to(asset)):
            return None
        if not path.is_file() or hashes.file(path) != info:
            return None
    assets = json.loads(manifest_path.read_text())["assets"]
    prefix = f"{dataset}/{scene}/"
    shared = {name: info for name, info in assets.items() if name.startswith(prefix)
              and (name[len(prefix):] in {"vocabulary.pt", "frames.pt", "metadata.json"} or name[len(prefix):].startswith("rgb/"))}
    if not {prefix + name for name in ("vocabulary.pt", "frames.pt", "metadata.json")}.issubset(shared):
        return None
    if not all((target / name).is_file() and hashes.file(target / name) == info for name, info in shared.items()):
        return None
    return json.loads(scene_path.read_text())


def save_changed_tensor(path: Path, value: Any) -> bool:
    if path.is_file():
        previous = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if isinstance(value, dict):
            unchanged = isinstance(previous, dict) and previous.keys() == value.keys() and all(
                previous[key].dtype == row.dtype and torch.equal(previous[key], row) for key, row in value.items())
        else:
            unchanged = isinstance(previous, torch.Tensor) and previous.dtype == value.dtype and torch.equal(previous, value)
        del previous
        if unchanged:
            return False
    save_tensor(path, value)
    return True


def prepare_scene(dataset: str, scene: str, asset: Path, source: Path, target: Path, mods: dict[str, Any], hashes: Hashes) -> dict[str, Any]:
    shared = reusable_shared_scene(dataset, scene, asset, source, target, hashes)
    if shared is not None:
        return shared
    destination = target / dataset / scene
    destination.mkdir(parents=True, exist_ok=True)
    provenance: list[Path] = []
    if dataset == "lerf_ovs":
        scoring = mods["lerf_scoring"]
        raw = asset / "lerf_ovs_raw" / "lerf_ovs"
        query_order = list(scoring.scene_queries(raw, scene))
        annotation_paths = scoring.annotation_files(raw, scene)
        annotations = [scoring.load_annotation(path, query_order) for path in annotation_paths]
        cameras = mods["lerf_gaussian"].load_cameras(raw / scene)
        image_root = raw / scene / "images"
        provenance.extend(annotation_paths)
        sparse = raw / scene / "sparse"
        provenance.extend(path for path in sparse.rglob("*") if path.is_file() and path.suffix in {".bin", ".txt"})
        camera_convention = "raw COLMAP/OpenCV world-to-camera; no recenter or NDC; annotation-resolution intrinsics"
    else:
        scoring = mods["ovs_scoring"]
        payload = mods["ovs_data"].load_scene(asset / "3dovs_raw", scene)
        query_order = list(payload["queries"])
        annotations, cameras = payload["annotations"], payload["cameras"]
        image_root = payload["scene_dir"] / "images"
        all_rgb = set(payload["image_paths"])
        provenance.extend(path for path in payload["source_files"] if path not in all_rgb)
        camera_convention = payload["camera_convention"] + "; " + payload["camera_source"]
    cache = source / WORKSPACES[dataset] / "text_cache.pt"
    vocab, text_metadata = vocabulary(cache, query_order, dataset)
    provenance.append(cache)
    frames = []
    for annotation in annotations:
        camera = scoring.camera_at_resolution(cameras[annotation.filename], annotation.width, annotation.height) if dataset == "lerf_ovs" else cameras[annotation.filename]
        rgb_source = image_root / annotation.filename
        if not rgb_source.is_file():
            raise FileNotFoundError(rgb_source)
        provenance.append(rgb_source)
        rgb_name = f"{annotation.filename}.png"
        rgb_path = destination / "rgb" / rgb_name
        rgb_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(rgb_source) as image:
            image = image.convert("RGB")
            if image.size != (annotation.width, annotation.height):
                image = image.resize((annotation.width, annotation.height), Image.Resampling.BILINEAR)
            image.save(rgb_path)
        masks = torch.from_numpy(np.stack([annotation.masks[name] for name in query_order])).to(torch.bool)
        frames.append({"filename": "rgb/" + rgb_name, "source_filename": annotation.filename,
                       "viewmat": torch.as_tensor(camera["viewmat"], dtype=torch.float32).clone(),
                       "K": torch.as_tensor(camera["K"], dtype=torch.float32).clone(),
                       "width": annotation.width, "height": annotation.height, "masks": masks})
    save_tensor(destination / "vocabulary.pt", vocab)
    save_tensor(destination / "frames.pt", {"frames": frames})
    metadata = {"dataset": dataset, "scene": scene, "frame_count": len(frames), "queries": query_order,
                "camera_convention": camera_convention,
                "annotation_policy": "rounded float32 polygon OpenCV fillPoly; scene query union sorted by (casefold,name)" if dataset == "lerf_ovs" else "official numeric annotation folder index into Python-sorted images; factor8 floor dimensions; nearest masks then (pixels/255).astype(int8)",
                "rgb_policy": "only annotated views; RGB PNG at evaluation resolution; bilinear resize when needed",
                "text_cache_metadata": text_metadata,
                "source_files": {str(path): hashes.file(path) for path in dict.fromkeys(provenance)},
                "converter": "tools/export_image_data.py", "schema_version": 1}
    save_json(destination / "metadata.json", metadata)
    return {"dataset": dataset, "scene": scene, "vocabulary": "vocabulary.pt", "frames": "frames.pt", "metadata": "metadata.json", "methods": {}, "schema_version": 1}


def load_case(dataset: str, scene: str, method: str, asset: Path, mods: dict[str, Any], device: str) -> Any:
    if dataset == "3d_ovs":
        return mods["ovs_gaussian"].load_scene(method, scene, asset, device)
    module = "lerf_gaussian" if method in {"sfs", "occam", "ludvig"} else "lerf_normvala" if method in {"normlift", "vala"} else "lerf_" + method
    return mods[module].load_scene(method, scene, asset / "lerf_ovs", device)


def export_case(dataset: str, scene: str, method: str, asset: Path, target: Path, mods: dict[str, Any], hashes: Hashes, device: str, chunk: int) -> dict[str, Any]:
    directory = target / dataset / scene / method
    metadata_path = directory / "metadata.json"
    runtime = numerical_runtime(device, chunk)
    input_identity = {"asset_root": str(asset.resolve()),
                      "source_modules": {name: str(Path(module.__file__).resolve()) for name, module in mods.items()},
                      "numerical_runtime": runtime}
    # Resume only fully hashed canonical files with matching archived source files.
    if metadata_path.is_file():
        previous = json.loads(metadata_path.read_text())
        files = previous.get("files", {})
        sources = previous.get("source_files", {})
        if previous.get("input_identity") == input_identity and files and sources and all((directory / name).is_file() and hashes.file(directory / name) == info for name, info in files.items()) and all(Path(path).is_file() and hashes.file(Path(path)) == info for path, info in sources.items()):
            return previous
    loaded = load_case(dataset, scene, method, asset, mods, device)
    scoring = mods["lerf_scoring" if dataset == "lerf_ovs" else "ovs_scoring"]
    features = scoring.normalize_scene_features(loaded.features, loaded.valid, chunk)
    geometry = {key: loaded._geometry[source_key].detach().to(device="cpu", dtype=torch.float32).contiguous()
                for key, source_key in {"means": "means", "scales": "scales", "quats": "quats", "opacity": "opacities"}.items()}
    valid = loaded.valid.detach().cpu().contiguous()
    count = len(features)
    directory.mkdir(parents=True, exist_ok=True)
    geometry_changed = save_changed_tensor(directory / "gaussians.pt", geometry)
    validity_changed = save_changed_tensor(directory / "feature_valid.pt", valid)
    save_tensor(directory / "features.pt", features.detach().cpu().contiguous())
    source_files = {}
    for key, value in loaded.metadata.items():
        if (key.endswith("_path") or key.endswith("_module")) and isinstance(value, str) and key != "base_export_path":
            path = Path(value)
            if path.is_file():
                source_files[str(path)] = hashes.file(path)
    for module in mods.values():
        path = Path(module.__file__)
        source_files[str(path)] = hashes.file(path)
    source_files[str(Path(__file__).resolve())] = hashes.file(Path(__file__))
    metadata = {"dataset": dataset, "scene": scene, "method": method, "relative": f"{dataset}/{scene}/{method}",
                "primitives": count, "feature_valid_count": int(valid.sum()),
                "input_identity": input_identity,
                "feature_normalization": f"authoritative {scoring.__name__}.normalize_scene_features; chunk={chunk}; device={device}; exactly once, invalid rows zero",
                "geometry_convention": "activated float32 geometry from authoritative adapter, original row order; no SH required",
                "conversion_runtime": {**runtime, "bit_exact_archived_gpu_claim": False},
                "geometry_rewritten": geometry_changed, "validity_rewritten": validity_changed,
                "image_eligibility": "adapter-valid only; no opacity threshold; invalid rows retain geometry and opacity",
                "adapter_metadata": loaded.metadata, "source_files": source_files,
                "files": {name: hashes.file(directory / name) for name in ("gaussians.pt", "features.pt", "feature_valid.pt")}, "schema_version": 1}
    save_json(metadata_path, metadata)
    del loaded, features, geometry, valid
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return metadata


def merge_manifests(target: Path, hashes: Hashes, failures: list[dict[str, str]], runtime: dict[str, Any]) -> dict[str, Any]:
    root_path = target / "manifest.json"
    root = json.loads(root_path.read_text()) if root_path.is_file() else {"schema_version": 1, "assets": {}, "cases": []}
    # Replace only image records. Point-cloud records/assets remain unchanged.
    root.setdefault("point_export_complete", bool(root.get("complete", True)) if "image_export" not in root else True)
    root["cases"] = [case for case in root.get("cases", []) if case["dataset"] not in SCENES]
    total_bytes = 0
    total_cases = 0
    total_frames = 0
    total_primitives = 0
    total_files = 0
    runtime_matched_cases = 0
    for dataset, scenes in SCENES.items():
        assets = {}
        entries = {}
        cases = []
        frame_count = 0
        for scene in scenes:
            scene_dir = target / dataset / scene
            scene_json = scene_dir / "scene.json"
            if not scene_json.is_file():
                continue
            contract = json.loads(scene_json.read_text())
            entries[scene] = {"scene": f"{scene}/scene.json", "methods": list(contract["methods"])}
            frame_count += json.loads((scene_dir / "metadata.json").read_text())["frame_count"]
            for method in contract["methods"]:
                case_dir = scene_dir / method
                metadata = json.loads((case_dir / "metadata.json").read_text())
                metadata["files"]["metadata.json"] = hashes.file(case_dir / "metadata.json")
                cases.append(metadata)
            for path in sorted(scene_dir.rglob("*")):
                if path.is_symlink():
                    raise RuntimeError(f"external symlink forbidden: {path}")
                if path.is_file() and not path.name.endswith(".tmp"):
                    assets[path.relative_to(target).as_posix()] = hashes.file(path)
        expected = len(scenes) * len(METHODS)
        matching_runtime = sum(case.get("input_identity", {}).get("numerical_runtime") == runtime for case in cases)
        manifest = {"dataset": dataset, "schema_version": 1, "scenes": entries, "cases": cases,
                    "case_count": len(cases), "expected_case_count": expected,
                    "frame_count": frame_count, "file_count": len(assets),
                    "primitive_count": sum(case["primitives"] for case in cases),
                    "required_numerical_runtime": runtime, "matching_runtime_case_count": matching_runtime,
                    "complete": len(cases) == expected and matching_runtime == expected and not any(failure["dataset"] == dataset for failure in failures),
                    "assets": assets, "total_bytes": sum(info["size"] for info in assets.values()),
                    "failures": [failure for failure in failures if failure["dataset"] == dataset]}
        save_json(target / dataset / "manifest.json", manifest)
        assets[f"{dataset}/manifest.json"] = hashes.file(target / dataset / "manifest.json")
        root.setdefault("assets", {})[dataset] = assets
        root["cases"].extend(cases)
        total_bytes += sum(info["size"] for info in assets.values())
        total_cases += len(cases)
        total_frames += frame_count
        total_primitives += manifest["primitive_count"]
        total_files += len(assets)
        runtime_matched_cases += matching_runtime
    root["case_count"] = len(root["cases"])
    root["image_export"] = {"case_count": total_cases, "expected_case_count": 98, "total_bytes": total_bytes,
                            "frame_count": total_frames, "primitive_count": total_primitives, "file_count": total_files,
                            "required_numerical_runtime": runtime, "matching_runtime_case_count": runtime_matched_cases,
                            "complete": total_cases == 98 and runtime_matched_cases == 98 and not failures, "failures": failures,
                            "schema": "per-scene scene.json mapping; frames.pt {'frames':list}; vocabulary.pt names/features/class_ids; activated gaussians; normalized features and explicit validity"}
    root["complete"] = root["point_export_complete"] and root["image_export"]["complete"]
    save_json(root_path, root)
    return root["image_export"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True, help="Archived evaluation_workspaces directory or its parent; conversion dependency only")
    parser.add_argument("--asset-root", type=Path, required=True, help="User-provided checkpoint/raw annotation bundle root")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--datasets", nargs="+", choices=tuple(SCENES), default=list(SCENES))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--device", default="cpu", help="One case at a time; CUDA can accelerate offline decoding")
    parser.add_argument("--chunk-size", type=int, default=16384)
    args = parser.parse_args()
    source = args.source_root.expanduser().resolve()
    if (source / "evaluation_workspaces").is_dir():
        source /= "evaluation_workspaces"
    asset, target = args.asset_root.expanduser().resolve(), args.output.expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("highest")
    if args.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    hashes = Hashes()
    failures = []
    for dataset in args.datasets:
        mods = modules(source, dataset)
        for scene in SCENES[dataset]:
            try:
                contract = prepare_scene(dataset, scene, asset, source, target, mods, hashes)
            except Exception as error:
                failures.append({"dataset": dataset, "scene": scene, "method": "shared", "error": f"{type(error).__name__}: {error}"})
                print(json.dumps(failures[-1]), flush=True)
                continue
            previous = target / dataset / scene / "scene.json"
            if previous.is_file():
                contract["methods"].update(json.loads(previous.read_text())["methods"])
            for method in args.methods:
                try:
                    metadata = export_case(dataset, scene, method, asset, target, mods, hashes, args.device, args.chunk_size)
                    contract["methods"][method] = {key: f"{method}/{filename}" for key, filename in {"gaussians": "gaussians.pt", "features": "features.pt", "feature_valid": "feature_valid.pt"}.items()}
                    save_json(previous, contract)
                    print(json.dumps({"exported": [dataset, scene, method], "primitives": metadata["primitives"], "valid": metadata["feature_valid_count"]}), flush=True)
                except Exception as error:
                    failures.append({"dataset": dataset, "scene": scene, "method": method, "error": f"{type(error).__name__}: {error}"})
                    print(json.dumps(failures[-1]), flush=True)
                    gc.collect()
                    if args.device.startswith("cuda"):
                        torch.cuda.empty_cache()
    summary = merge_manifests(target, hashes, failures, numerical_runtime(args.device, args.chunk_size))
    print(json.dumps(summary, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
