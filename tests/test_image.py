"""Image protocol regressions without CUDA, gsplat, archives, or CLIP weights."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from eval.image import (
    THRESHOLDS, ImageScene, binary_curve_counts, count_metrics, evaluate_case,
    load_scene, load_method, loso, multiclass_counts, scene_macro,
)


class ImageBehaviorTests(unittest.TestCase):
    def test_lerf_ignores_background_overlap_and_rejects_unsupported(self):
        masks = np.asarray([[[True, False, True, False]], [[False, True, True, False]]])
        labels = np.asarray([[0, -1, 1, 0]], dtype=np.int32)
        queries, counts = multiclass_counts("lerf_ovs", labels, masks)
        self.assertEqual(queries.tolist(), [0, 1])
        self.assertEqual(counts.tolist(), [[1, 0, 0, 1], [0, 0, 1, 1]])
        self.assertEqual(scene_macro("lerf_ovs", "multiclass", queries, counts).tolist(), [0.5, 0.5, 0.5])

    def test_ovs_includes_absent_classes_background_and_overlap(self):
        masks = np.asarray([[[True, False, True, False]], [[False, False, True, False]], [[False, False, False, False]]])
        labels = np.asarray([[0, 0, 0, 0]], dtype=np.int32)
        queries, counts = multiclass_counts("3d_ovs", labels, masks)
        self.assertEqual(queries.tolist(), [0, 1, 2])
        self.assertEqual(counts.tolist(), [[2, 2, 0, 0], [0, 0, 1, 3], [0, 0, 0, 4]])
        self.assertAlmostEqual(scene_macro("3d_ovs", "multiclass", queries, counts)[0], 1 / 6)

    def test_binary_full_image_negatives_unsupported_false_negatives_and_ge(self):
        masks = np.asarray([[[True, True, False, False]], [[False, False, False, False]]])
        scores = np.asarray([[[0.5, 1], [0.9, 1], [0.5, 1], [0.9, 1]]], dtype=np.float32)
        support = np.asarray([[True, False, True, False]])
        for dataset in ("lerf_ovs", "3d_ovs"):
            queries, counts = binary_curve_counts(dataset, scores, masks, support)
            self.assertEqual(queries.tolist(), [0])
            index = int(np.flatnonzero(THRESHOLDS == 0.5)[0])
            self.assertEqual(counts[0, index].tolist(), [1, 1, 1, 1])
            self.assertEqual(counts[0, -1].tolist(), [0, 0, 2, 2])

    def test_lerf_and_ovs_preserve_authoritative_threshold_float_arithmetic(self):
        scores = np.asarray([[[0.7]]], dtype=np.float32)
        masks = np.ones((1, 1, 1), dtype=bool)
        support = np.ones((1, 1), dtype=bool)
        index = int(np.flatnonzero(THRESHOLDS == 0.7)[0])
        # float32(0.7) is below float64(0.7); LERF compares float32 scalars,
        # OVS casts sorted pixel scores to float64 before comparison.
        self.assertEqual(binary_curve_counts("lerf_ovs", scores, masks, support)[1][0, index, 0], 1)
        self.assertEqual(binary_curve_counts("3d_ovs", scores, masks, support)[1][0, index, 0], 0)

    def test_nested_present_query_macro_not_frame_pair_pooling(self):
        queries = np.asarray([0, 0, 1])
        counts = np.asarray([[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0]])
        self.assertEqual(scene_macro("lerf_ovs", "multiclass", queries, counts).tolist(), [0.5, 0.5, 0.5])
        self.assertEqual(scene_macro("3d_ovs", "single_query", queries, counts).tolist(), [0.5, 0.5, 0.5])
        self.assertAlmostEqual(scene_macro("3d_ovs", "multiclass", queries, counts)[0], 2 / 3)

    def test_loso_never_uses_heldout_objective_and_smallest_tie(self):
        curves = {name: np.zeros((201, 3)) for name in ("a", "b", "c", "d")}
        curves["a"][180] = 1
        for name in ("b", "c", "d"):
            curves[name][20:22] = 0.7
        metrics, folds = loso(curves, tuple(curves))
        self.assertEqual(folds["a"]["selected_threshold"], -0.8)
        self.assertEqual(folds["a"]["training_scenes"], ["b", "c", "d"])
        self.assertEqual(folds["a"]["heldout_metrics_percent"]["miou"], 0)
        self.assertAlmostEqual(metrics[0], 0.525)
        with self.assertRaises(ValueError):
            loso({"a": curves["a"]}, ("a", "b"))

    def test_zero_denominators_are_zero_not_nan(self):
        np.testing.assert_array_equal(count_metrics(np.asarray([[0, 0, 0, 10]])), [[0, 0, 0]])

    def test_portable_loader_does_not_renormalize_or_read_rgb(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary)
            directory = data / "lerf_ovs" / "figurines"
            (directory / "sfs").mkdir(parents=True)
            manifest = {"dataset": "lerf_ovs", "scene": "figurines", "vocabulary": "vocabulary.pt", "frames": "frames.pt", "methods": {"sfs": {"gaussians": "sfs/gaussians.pt", "features": "sfs/features.pt", "feature_valid": "sfs/feature_valid.pt"}}}
            (directory / "scene.json").write_text(json.dumps(manifest))
            text = torch.zeros(1, 512)
            text[0, 0] = 1
            torch.save({"names": ["object"], "features": text, "class_ids": torch.tensor([7])}, directory / "vocabulary.pt")
            torch.save({"frames": [{"filename": "rgb/not-required.png", "viewmat": torch.eye(4), "K": torch.eye(3), "width": 1, "height": 1, "masks": torch.ones(1, 1, 1, dtype=torch.bool)}]}, directory / "frames.pt")
            features = torch.zeros(2, 512)
            features[0, 0] = 0.01
            geometry = {"means": torch.zeros(2, 3), "scales": torch.ones(2, 3), "quats": torch.tensor([[1., 0, 0, 0]] * 2), "opacity": torch.tensor([0.0001, 0.9])}
            torch.save(geometry, directory / "sfs/gaussians.pt")
            torch.save(features, directory / "sfs/features.pt")
            torch.save(torch.tensor([True, False]), directory / "sfs/feature_valid.pt")
            scene = load_scene(data, "lerf_ovs", "figurines")
            loaded_geometry, loaded_features, valid = load_method(scene, "sfs")
            self.assertTrue(torch.equal(loaded_features, features))
            self.assertTrue(torch.equal(loaded_geometry["opacity"], geometry["opacity"]))
            self.assertEqual(valid.tolist(), [True, False])

    def test_mixed_case_vocabulary_preserves_dataset_specific_row_alignment(self):
        names = ["spoon", "Stainless steel pots", "toaster"]
        text = torch.zeros(3, 512)
        text[torch.arange(3), torch.arange(3)] = 1
        masks = torch.eye(3, dtype=torch.bool).reshape(3, 1, 3)
        ids = torch.tensor([17, 42, 9])
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary)
            for dataset, scene_name in (("lerf_ovs", "waldo_kitchen"), ("3d_ovs", "snacks")):
                directory = data / dataset / scene_name
                directory.mkdir(parents=True)
                manifest = {"dataset": dataset, "scene": scene_name, "vocabulary": "vocabulary.pt", "frames": "frames.pt", "methods": {}}
                (directory / "scene.json").write_text(json.dumps(manifest))
                torch.save({"names": names, "features": text, "class_ids": ids}, directory / "vocabulary.pt")
                torch.save({"frames": [{"filename": "rgb/not-required.png", "viewmat": torch.eye(4), "K": torch.eye(3), "width": 3, "height": 1, "masks": masks}]}, directory / "frames.pt")
                if dataset == "lerf_ovs":
                    scene = load_scene(data, dataset, scene_name)
                    self.assertEqual(scene.names, names)
                    self.assertTrue(torch.equal(scene.text, text))
                    self.assertTrue(torch.equal(scene.class_ids, ids))
                    self.assertTrue(torch.equal(scene.frames[0]["masks"], masks))
                    # Lexically sorting names alone would silently change channel
                    # meaning. Reject that producer order rather than repairing it.
                    torch.save({"names": sorted(names), "features": text, "class_ids": ids}, directory / "vocabulary.pt")
                    with self.assertRaisesRegex(ValueError, "canonical lerf_ovs ordering"):
                        load_scene(data, dataset, scene_name)
                else:
                    with self.assertRaisesRegex(ValueError, "canonical 3d_ovs ordering"):
                        load_scene(data, dataset, scene_name)
                    order = torch.tensor([1, 0, 2])
                    torch.save({"names": sorted(names), "features": text[order], "class_ids": ids[order]}, directory / "vocabulary.pt")
                    torch.save({"frames": [{"filename": "rgb/not-required.png", "viewmat": torch.eye(4), "K": torch.eye(3), "width": 3, "height": 1, "masks": masks[order]}]}, directory / "frames.pt")
                    scene = load_scene(data, dataset, scene_name)
                    self.assertEqual(scene.names, sorted(names))
                    self.assertTrue(torch.equal(scene.text, text[order]))
                    self.assertTrue(torch.equal(scene.class_ids, ids[order]))
                    self.assertTrue(torch.equal(scene.frames[0]["masks"], masks[order]))

    def test_case_eligibility_keeps_low_opacity_and_all_geometry_occluders(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "scene.json").write_text("{}")
            for name in ("v.pt", "f.pt", "g.pt", "x.pt", "valid.pt"):
                (directory / name).write_bytes(b"provenance")
            features = torch.zeros(3, 512)
            features[0, 0] = 1
            features[1, 1] = 1
            geometry = {"means": torch.zeros(3, 3), "quats": torch.ones(3, 4), "scales": torch.ones(3, 3), "opacity": torch.tensor([0.00001, 0.8, 0.9])}
            frame = {"filename": "rgb/example.png", "viewmat": torch.eye(4), "K": torch.eye(3), "width": 2, "height": 1, "masks": torch.tensor([[[True, False]], [[False, True]]])}
            paths = {"gaussians": "g.pt", "features": "x.pt", "feature_valid": "valid.pt"}
            manifest = {"scene": "figurines", "vocabulary": "v.pt", "frames": "f.pt"}
            scene = ImageScene(directory, ["a", "b"], features[:2], torch.tensor([0, 1]), [frame], {"sfs": paths}, manifest)
            calls = []
            test = self

            class FakeRenderer:
                def __init__(self, actual_geometry, device):
                    test.assertEqual(len(actual_geometry["means"]), 3)

                def render(self, values, camera):
                    test.assertEqual(values[2].tolist(), [0.] * values.shape[1])
                    return values[:2].numpy()[None]

            def scoring(statistics, eligible, text, **options):
                test.assertEqual(eligible.tolist(), [True, True, False])
                test.assertTrue(torch.equal(statistics.features, features))
                calls.append(options)
                return torch.tensor([[1., 0.], [0., 1.], [-torch.inf, -torch.inf]]), {}

            with patch("eval.image.load_method", return_value=(geometry, features, torch.tensor([True, True, False]))), patch("eval.image.SemanticRenderer", FakeRenderer), patch("eval.image.score_embeddings", scoring):
                result = evaluate_case(scene, "lerf_ovs", "sfs", ["multiclass", "single_query"], torch.device("cpu"), directory / "output")
            self.assertEqual([(call["mode"], call["raw"]) for call in calls], [("multiclass", True), ("multiclass", False), ("single_query", True), ("single_query", False)])
            self.assertEqual(result["multiclass"]["raw"].tolist(), [1., 1., 1.])
            self.assertTrue((directory / "output/single_query_calibrated_counts.npz").is_file())


if __name__ == "__main__":
    unittest.main()
