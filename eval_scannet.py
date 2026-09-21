"""Evaluate the fixed OpenGaussian19 ScanNet vocabulary with exact P3 correspondence."""

from evaluation import DatasetVocabularySpec, run_dataset_cli


SPEC = DatasetVocabularySpec(
    dataset="scannet",
    class_ids=tuple(range(19)),
    names=(
        "wall",
        "floor",
        "cabinet",
        "bed",
        "chair",
        "sofa",
        "table",
        "door",
        "window",
        "bookshelf",
        "picture",
        "counter",
        "desk",
        "curtain",
        "refrigerator",
        "shower curtain",
        "toilet",
        "sink",
        "bathtub",
    ),
    subsets={
        "15": (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 16, 17),
        "10": (0, 1, 3, 4, 5, 6, 7, 8, 9, 16),
    },
)


if __name__ == "__main__":
    run_dataset_cli(SPEC)
