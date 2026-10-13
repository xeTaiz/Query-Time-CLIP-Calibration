"""Consumer-visible point support, correspondence, and aggregation regressions."""
import unittest

import torch

from eval.evaluation import assign_mahalanobis, evaluate_scores
from eval.point import aggregate_results


def geometry(means, scales):
    count = len(means)
    return {
        "means": torch.tensor(means, dtype=torch.float32),
        "scales": torch.tensor(scales, dtype=torch.float32),
        "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]] * count),
        "opacity": torch.ones(count),
    }


class PointBehaviorTests(unittest.TestCase):
    def test_correspondence_is_mahalanobis_not_nearest_center(self):
        gaussians = geometry([[0, 0, 0], [2, 0, 0]], [[0.1, 0.1, 0.1], [5, 5, 5]])
        assigned = assign_mahalanobis(
            torch.tensor([[0.1, 1.0, 0.0]]), gaussians, torch.tensor([True, True]), device="cpu"
        )
        self.assertEqual(assigned.tolist(), [1])

    def test_subset_keeps_other_labels_as_binary_negatives(self):
        gaussians = geometry([[0, 0, 0], [1, 0, 0]], [[1, 1, 1], [1, 1, 1]])
        features = torch.zeros(2, 512)
        features[:, 0] = 1
        points = torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [1.0, 0, 0]])
        labels = torch.tensor([0, 1, -1])
        scores = torch.tensor([[0.5], [0.5]])
        assignments = torch.tensor([0, 1, 1])
        binary = evaluate_scores(
            gaussians, features, scores, points, labels, torch.tensor([0]),
            mode="single_query", threshold=0.2, assignments=assignments, device="cpu",
        )
        self.assertEqual([binary["per_class"][0][key] for key in ("tp", "fp", "fn")], [1, 1, 0])
        self.assertEqual(binary["miou"], 50.0)
        self.assertEqual(binary["provenance"]["population"]["evaluated_gt_points"], 2)
        multiclass = evaluate_scores(
            gaussians, features, scores, points, labels, torch.tensor([0]),
            assignments=assignments, device="cpu",
        )
        self.assertEqual(multiclass["miou"], 100.0)
        self.assertEqual(multiclass["provenance"]["population"]["evaluated_gt_points"], 1)

    def test_dataset_macro_weights_scenes_not_points(self):
        cases = []
        for scene, value, tp, fn in (("small", 100.0, 1, 0), ("large", 0.0, 0, 100)):
            cases.append({
                "dataset": "scannet", "subset": "full", "method": "sfs", "task": "single_query",
                "variant": "raw", "scene": scene, "miou": value, "precision": value, "recall": value,
                "per_class": [{"class_id": 0, "class_name": "wall", "tp": tp, "fp": 0, "fn": fn,
                               "iou": value, "precision": value, "recall": value}],
            })
        aggregate = aggregate_results(cases)[0]
        self.assertEqual(aggregate["miou"], 50.0)
        self.assertEqual(aggregate["per_class"][0]["iou"], 50.0)
        self.assertEqual(aggregate["per_class"][0]["total_fn"], 100)


if __name__ == "__main__":
    unittest.main()
