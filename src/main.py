import datetime
import os
import random
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
from fire import Fire

from detect import BubbleDetector
from load import BubbleAnnotation, load_annotations, load_wav
from plot import plot_annotations, plot_wav
from sample import sample_training_data

SAMPLE_RATE = 44100
WINDOW_SIZE = 0.1  # seconds
TRAIN_RATIO = 0.7
MERGE_THRESHOLD = 10  # in samples

annotations = load_annotations("data/annotations.json")


def prepare_data(file: Optional[str] = None):
    if file is not None:
        wav_files = [file]
    else:
        wav_files = [f for f in os.listdir("data/") if f.endswith(".wav")]

    print(f"Found {len(wav_files)} wav files")

    audio_files = []
    offsets = {}
    offset = 0
    for file_name in wav_files:
        _, audio = load_wav(f"data/{file_name}")
        audio = audio.astype(float)
        offsets[file_name] = offset
        offset += audio.size
        audio_files.append(audio)

    data = np.concatenate(audio_files)

    all_annotations = []
    for file_name in annotations:
        if file_name not in offsets:
            continue
        all_annotations.extend(
            [
                BubbleAnnotation(start=ann[0] + offsets[file_name], end=ann[1] + offsets[file_name])
                for ann in annotations[file_name]
            ]
        )

    pos, neg = sample_training_data(
        len(data),
        all_annotations,
        window_size=int(WINDOW_SIZE * SAMPLE_RATE),
        count=1000,
    )

    print(f"Obtained {len(pos)}+ and {len(neg)}- samples total.")

    # Split into train and test sets
    n_pos_train = int(TRAIN_RATIO * len(pos))
    n_neg_train = int(TRAIN_RATIO * len(neg))

    shuffled_indices_pos = list(range(len(pos)))
    random.shuffle(shuffled_indices_pos)
    shuffled_indices_neg = list(range(len(neg)))
    random.shuffle(shuffled_indices_neg)

    train_positive = [pos[i] for i in shuffled_indices_pos[:n_pos_train]]
    test_positive = [pos[i] for i in shuffled_indices_pos[n_pos_train:]]

    train_negative = [neg[i] for i in shuffled_indices_neg[:n_neg_train]]
    test_negative = [neg[i] for i in shuffled_indices_neg[n_neg_train:]]

    return (
        data,
        train_positive,
        train_negative,
        test_positive,
        test_negative,
    )


def visualize_waveform(file_name):
    t, audio = load_wav(f"data/{file_name}")
    audio = audio.astype(float)

    fig, ax = plt.subplots(figsize=(12, 4))
    plot_wav(ax, t, audio, title=file_name)

    base_name = os.path.basename(file_name)
    if base_name in annotations:
        plot_annotations(ax, annotations[base_name], audio)

    plt.show()


def visualize_detections(detector: BubbleDetector, data, annotations, file_name):
    fig, (ax_gt, ax_det) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, sharey=True)

    t = np.arange(len(data)) / SAMPLE_RATE

    # Annotations (ground truth)
    plot_wav(ax_gt, t, data, title=f"{file_name} annotations (ground truth)")
    base_name = os.path.basename(file_name)
    if base_name in annotations:
        plot_annotations(
            ax_gt,
            annotations[base_name],
            data,
            color="green",
            label="Annotations",
        )
    ax_gt.legend()

    # Detections
    plot_wav(ax_det, t, data, title=f"{file_name} detections")

    window = int(WINDOW_SIZE * SAMPLE_RATE)
    intervals = [
        BubbleAnnotation(start=i, end=i + window) for i in range(0, len(data) - window, window)
    ]

    transformed = detector.preprocessor.transform(data)
    transformed_intervals = [
        detector.preprocessor.transform_interval(interval) for interval in intervals
    ]
    labels = detector.detect(transformed, transformed_intervals)
    detected_intervals = [interval for interval, label in zip(intervals, labels) if label]

    if detected_intervals:
        # merge neighbouring intervals
        merged_intervals = detected_intervals[:1]
        for interval in detected_intervals[1:]:
            if np.abs(interval.start - merged_intervals[-1].end) <= MERGE_THRESHOLD:
                merged_intervals[-1] = BubbleAnnotation(
                    start=merged_intervals[-1].start,
                    end=interval.end,
                )
            else:
                merged_intervals.append(interval)

        plot_annotations(
            ax_det,
            merged_intervals,
            data,
            color="red",
            label="Detections",
        )
    else:
        print("No bubbles detected.")

    ax_det.legend()
    ax_det.set_xlabel("Time (s)")

    plt.tight_layout()
    plt.show()


class Main:
    def visualize_waveform(self, file: str):
        visualize_waveform(file)

    def train_detector(
        self,
        name: str = "all",
        preprocessor: str = "identity",
        file: Optional[str] = None,
        save: bool = False,
    ):
        """Train and evaluate a bubble detector."""
        available_models = BubbleDetector.detectors.keys()
        available_preprocessors = BubbleDetector.preprocessors.keys()

        if name not in list(available_models) + ["all"]:
            raise ValueError(f"Available models: {list(available_models)}")
        if preprocessor not in available_preprocessors:
            raise ValueError(f"Available preprocessors: {list(available_preprocessors)}")

        data, train_positive, train_negative, test_positive, test_negative = prepare_data()

        print(
            f"Loaded data of total length {data.shape} samples ({data.shape[0]/SAMPLE_RATE:.0f} seconds)"
        )
        print(f"Using window size of {WINDOW_SIZE}s ({int(WINDOW_SIZE * SAMPLE_RATE)} samples)")

        if name == "all":
            for model_name in available_models:
                detector = BubbleDetector(model_name, preprocessor)
                detector.train(
                    data=data,
                    positive_intervals=train_positive,
                    negative_intervals=train_negative,
                )
                detector.evaluate(
                    data=data,
                    positive_intervals=test_positive,
                    negative_intervals=test_negative,
                )
        else:
            detector = BubbleDetector(name, preprocessor)
            detector.train(
                data=data,
                positive_intervals=train_positive,
                negative_intervals=train_negative,
            )
            if save:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                detector.save(f"models/{name}_{preprocessor}_{timestamp}.model")
            if file is None:
                detector.evaluate(
                    data=data,
                    positive_intervals=test_positive,
                    negative_intervals=test_negative,
                )
            else:
                data = prepare_data(file)
                visualize_detections(
                    detector=detector,
                    data=data[0],
                    annotations=annotations,
                    file_name=file,
                )

    def eval_detector(self, model_file: str, file: Optional[str] = None):
        """Evaluate a trained bubble detector."""
        detector = joblib.load(model_file)

        if file is None:
            data, _, _, test_positive, test_negative = prepare_data()
            detector.evaluate(
                data=data,
                positive_intervals=test_positive,
                negative_intervals=test_negative,
            )
        else:
            data = prepare_data(file)
            visualize_detections(
                detector=detector,
                data=data[0],
                annotations=annotations,
                file_name=file,
            )


if __name__ == "__main__":
    Fire(Main)
