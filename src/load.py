import json
import os
from typing import NamedTuple

import numpy as np
from scipy.io import wavfile


class BubbleAnnotation(NamedTuple):
    """Represents a bubble annotation with start and end sample indices."""

    start: int
    end: int


def load_annotations(file_path: os.Path) -> dict[str, list[BubbleAnnotation]]:
    """Load bubble annotations from a Label Studio JSON file."""
    # Label studio JSON format
    with open(file_path, "r") as f:
        annotations = json.load(f)

    bubble_annotations = dict()
    for annotation in annotations:
        s = "-".join(annotation["audio"].split("/")[-1].split("-")[1:])
        file_name = s[:-10]
        file_name += ":".join(s[-10:-6][i : i + 2] for i in range(0, 6, 2))
        file_name += s[-6:]

        bubble_ranges = []

        for label in annotation["label"]:
            if "Bubbles" in label["labels"]:
                start = int(label["start"] * 44100)
                end = int(label["end"] * 44100)
                bubble_ranges.append((start, end))

        bubble_annotations[file_name] = bubble_ranges
    return bubble_annotations


def load_wav(file_path: os.PathLike):
    """Read a WAV file and return time and audio arrays."""
    fs, audio = wavfile.read(file_path)
    if audio.ndim > 1:
        print("Mixing to mono.")
        audio = audio.mean(axis=1)
    t = np.arange(len(audio)) / fs
    return t, audio
