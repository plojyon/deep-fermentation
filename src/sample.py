import bisect
import random

import numpy as np

from load import BubbleAnnotation


def get_sample(data: np.ndarray, interval: BubbleAnnotation):
    """Extract sample from data given an interval."""
    s, e = interval.start, interval.end

    # handle edge cases
    if interval.end - interval.start == 0:
        if interval.end >= data.shape[1]:
            s -= 1
        else:
            e += 1

    sample = data[:, s:e]
    if sample.shape[0] <= 100:
        sample = sample.flatten()
    else:
        sample = sample.mean(axis=1)
        # sample = sample[:, 0]
    return sample


def invert_intervals(intervals: BubbleAnnotation, total_length: int):
    """Invert a list of intervals."""
    inverted = []
    prev_end = 0
    for start, end in intervals:
        if start > prev_end:  # handle zero-width gaps
            inverted.append((prev_end, start))

        prev_end = end

    if prev_end < total_length:  # final interval
        inverted.append((prev_end, total_length))

    return inverted


def sample_one(legal_sampling_ranges: list[BubbleAnnotation]) -> list[BubbleAnnotation]:
    """Sample a number within given intervals."""

    # Build cumulative lengths
    lengths = []
    total = 0
    for start, end in legal_sampling_ranges:
        total += end - start
        lengths.append(total)

    r = random.randint(0, total)

    # find the sampled range
    idx = bisect.bisect_left(lengths, r)
    start, end = legal_sampling_ranges[idx]

    # calculate offset
    prev = lengths[idx - 1] if idx > 0 else 0

    return start + (r - prev)


def sample_within_annotations(
    intervals: list[BubbleAnnotation],
    window_size: int,
    count: int,
) -> list[BubbleAnnotation]:
    """Sample non-overlapping fixed-size windows within given intervals."""
    if not intervals:
        print("No annotations to sample from! Wtf?")
        return []

    # build initial legal sampling ranges
    legal_sampling_ranges = []
    for start, end in intervals:
        if end - start >= window_size:
            legal_sampling_ranges.append((start, end - window_size))

    samples = []
    while len(samples) < count:
        if not legal_sampling_ranges:
            print(f"Dead end, cannot sample anything else with window size {window_size}")
            print(f"Obtained {len(samples)} samples out of requested {count}.")
            break

        sample = sample_one(legal_sampling_ranges)
        samples.append(sample)

        # update legal sampling ranges
        new_ranges = []
        for range_start, range_end in legal_sampling_ranges:
            if sample + window_size <= range_start or sample - window_size >= range_end:
                # no overlap
                new_ranges.append((range_start, range_end))
            else:
                # overlap, split if necessary
                if range_start < sample - window_size:
                    new_ranges.append((range_start, sample - window_size))
                if sample + window_size < range_end:
                    new_ranges.append((sample + window_size, range_end))
        legal_sampling_ranges = new_ranges

    return [BubbleAnnotation(start=sample, end=sample + window_size) for sample in samples]


def sample_training_data(
    length: int,
    annotations: list[BubbleAnnotation],
    window_size: int,  # in samples
    count: int,
) -> tuple[list[BubbleAnnotation], list[BubbleAnnotation]]:
    """Extract fixed-size windows from audio.

    Windows have either full or no overlap with bubbles and no overlap between
    each other.
    """
    positive_windows = sample_within_annotations(annotations, window_size, count)

    inverted_annotations = invert_intervals(annotations, length)
    negative_windows = sample_within_annotations(
        inverted_annotations, window_size, min(count, len(positive_windows))
    )

    return positive_windows, negative_windows
