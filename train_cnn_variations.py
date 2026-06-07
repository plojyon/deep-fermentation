from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from src.detect import BubbleDetector
from src.main import SAMPLE_RATE, prepare_data

OUTPUT_DIR = Path("models/cnn_variations_big_kernel_identity")

# best:
# identity:baseline__ch16x32x64__ks3__hd64__do025__lr1e-3__bs32__ep20__norm1__bal1 (0.89)
# stft:larger_kernel__ch16x32x64__ks5__hd64__do025__lr1e-3__bs32__ep20__norm1__bal1 (0.75 vs 0.71 base)

KERNEL_SIZE_BASE = 2

VARIATIONS: list[dict[str, object]] = [
    {
        "tag": "baseline__ch16x32x64__ks3__hd64__do025__lr1e-3__bs32__ep20__norm1__bal1",
        "model_parameters": {
            "conv_channels": (16, 32, 64),
            "kernel_size": int(KERNEL_SIZE_BASE * 3),
            "hidden_dim": 64,
            "dropout": 0.15,
            "learning_rate": 1e-3,
            "batch_size": 32,
            "epochs": 20,
            "normalize": True,
            "balance_classes": True,
        },
    },
    {
        "tag": "shallow__ch8x16x32__ks3__hd64__do025__lr1e-3__bs32__ep20__norm1__bal1",
        "model_parameters": {
            "conv_channels": (8, 16, 32),
            "kernel_size": int(KERNEL_SIZE_BASE * 3),
            "hidden_dim": 64,
            "dropout": 0.15,
            "learning_rate": 1e-3,
            "batch_size": 32,
            "epochs": 20,
            "normalize": True,
            "balance_classes": True,
        },
    },
    {
        "tag": "wide__ch32x64x128__ks3__hd64__do025__lr1e-3__bs32__ep20__norm1__bal1",
        "model_parameters": {
            "conv_channels": (32, 64, 128),
            "kernel_size": int(KERNEL_SIZE_BASE * 3),
            "hidden_dim": 64,
            "dropout": 0.15,
            "learning_rate": 1e-3,
            "batch_size": 32,
            "epochs": 20,
            "normalize": True,
            "balance_classes": True,
        },
    },
    {
        "tag": "larger_kernel__ch16x32x64__ks5__hd64__do025__lr1e-3__bs32__ep20__norm1__bal1",
        "model_parameters": {
            "conv_channels": (16, 32, 64),
            "kernel_size": int(KERNEL_SIZE_BASE * 5),
            "hidden_dim": 64,
            "dropout": 0.15,
            "learning_rate": 1e-3,
            "batch_size": 32,
            "epochs": 20,
            "normalize": True,
            "balance_classes": True,
        },
    },
    {
        "tag": "smaller_hidden__ch16x32x64__ks3__hd32__do025__lr1e-3__bs32__ep20__norm1__bal1",
        "model_parameters": {
            "conv_channels": (16, 32, 64),
            "kernel_size": int(KERNEL_SIZE_BASE * 3),
            "hidden_dim": 32,
            "dropout": 0.15,
            "learning_rate": 1e-3,
            "batch_size": 32,
            "epochs": 20,
            "normalize": True,
            "balance_classes": True,
        },
    },
    {
        "tag": "larger_hidden__ch16x32x64__ks3__hd128__do025__lr1e-3__bs32__ep20__norm1__bal1",
        "model_parameters": {
            "conv_channels": (16, 32, 64),
            "kernel_size": int(KERNEL_SIZE_BASE * 3),
            "hidden_dim": 128,
            "dropout": 0.15,
            "learning_rate": 1e-3,
            "batch_size": 32,
            "epochs": 20,
            "normalize": True,
            "balance_classes": True,
        },
    },
    {
        "tag": "lower_lr__ch16x32x64__ks3__hd64__do025__lr3e-4__bs32__ep20__norm1__bal1",
        "model_parameters": {
            "conv_channels": (16, 32, 64),
            "kernel_size": int(KERNEL_SIZE_BASE * 3),
            "hidden_dim": 64,
            "dropout": 0.15,
            "learning_rate": 3e-4,
            "batch_size": 32,
            "epochs": 20,
            "normalize": True,
            "balance_classes": True,
        },
    },
]


def format_metric(value: float) -> str:
    return f"{value:.3f}"


def main() -> None:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data, train_positive, train_negative, test_positive, test_negative = prepare_data()
    print(
        f"Loaded data of total length {data.shape} samples "
        f"({data.shape[0] / SAMPLE_RATE:.0f} seconds)"
    )

    summary: list[dict[str, object]] = []
    for index, variation in enumerate(VARIATIONS, start=1):
        tag = str(variation["tag"])
        model_parameters = dict(variation["model_parameters"])

        print(f"[{index}/{len(VARIATIONS)}] Training {tag}")
        detector = BubbleDetector("cnn", "identity", model_parameters=model_parameters)
        detector.train(
            data=data,
            positive_intervals=train_positive,
            negative_intervals=train_negative,
        )
        precision, recall, f1 = detector.evaluate(
            data=data,
            positive_intervals=test_positive,
            negative_intervals=test_negative,
            to_stdout=False,
        )

        model_path = OUTPUT_DIR / f"{tag}.model"
        detector.save(str(model_path))

        print(
            f"  saved {model_path.name} | "
            f"precision={format_metric(precision)} "
            f"recall={format_metric(recall)} "
            f"f1={format_metric(f1)}"
        )

        summary.append(
            {
                "tag": tag,
                "model_path": str(model_path),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "model_parameters": model_parameters,
            }
        )

    summary_path = OUTPUT_DIR / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    best = max(summary, key=lambda item: float(item["f1"]))
    print(f"Best F1: {format_metric(float(best['f1']))} from {best['tag']}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
