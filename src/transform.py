import bisect
import random

import numpy as np

from load import BubbleAnnotation


def hann_window(length: int) -> np.ndarray:
    """Generate a Hann window of given length."""
    n = np.arange(length)
    window = 0.5 * (1 - np.cos(2 * np.pi * n / (length - 1)))
    return window


def stft(segment: np.ndarray):
    """Compute STFT of a segment and return magnitude in dB."""
    # STFT parameters
    n_fft = 2048
    hop = n_fft // 4
    win = hann_window(n_fft)

    # pad segment so it fits an integer number of hops
    if segment.size < n_fft:
        # pad to at least one frame
        pad_len = n_fft - segment.size
        segment_padded = np.pad(segment, (0, pad_len))
        num_frames = 1
    else:
        num_frames = 1 + int(np.ceil((segment.size - n_fft) / hop))
        total_len = n_fft + (num_frames - 1) * hop
        pad_len = total_len - segment.size
        segment_padded = np.pad(segment, (0, pad_len))

    # frame and compute FFT
    frames = np.lib.stride_tricks.as_strided(
        segment_padded,
        shape=(num_frames, n_fft),
        strides=(segment_padded.strides[0] * hop, segment_padded.strides[0]),
    ).copy()
    frames *= win[np.newaxis, :]

    # shape (freq_bins, time_frames)
    stft = np.fft.rfft(frames, n=n_fft, axis=1).T
    mag = np.abs(stft)
    mag_db = 20 * np.log10(mag + 1e-10)

    return mag_db


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


def sample_within_annotations(
    audio: np.ndarray,
    intervals: list[BubbleAnnotation],
    window_size: int,
    count: int,
):
    """Sample fixed-size windows within given intervals."""
    legal_sampling_ranges = []
    for start, end in intervals:
        if end - start >= window_size:
            legal_sampling_ranges.append((start, end - window_size))

    # Build cumulative lengths
    lengths = []
    total = 0
    for start, end in intervals:
        total += end - start
        lengths.append(total)

    samples = []
    for _ in range(count):
        r = random.randint(0, total)

        # find the sampled interval
        idx = bisect.bisect_left(lengths, r)
        start, end = intervals[idx]

        # calculate offset
        prev = lengths[idx - 1] if idx > 0 else 0
        samples.append(start + (r - prev))

    return [
        audio[sample : sample + window_size]  # noqa: E203
        for sample in samples
        # burek
    ]


def sample_training_data(
    audio: np.ndarray,
    annotations: list[BubbleAnnotation],
    window_size: int,  # in samples
    count_positive: int,
    count_negative: int,
):
    """Extract fixed-size windows from audio.

    Windows have either full or no overlap with bubbles.
    """
    positive_windows = sample_within_annotations(
        audio, annotations, window_size, count_positive
    )

    inverted_annotations = invert_intervals(annotations, len(audio))
    negative_windows = sample_within_annotations(
        audio, inverted_annotations, window_size, count_negative
    )

    return positive_windows, negative_windows
