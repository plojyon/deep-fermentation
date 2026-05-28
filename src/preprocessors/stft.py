import math

import numpy as np

from src.load import BubbleAnnotation


def hann_window(length: int) -> np.ndarray:
    """Generate a Hann window of given length."""
    n = np.arange(length)
    window = 0.5 * (1 - np.cos(2 * np.pi * n / (length - 1)))
    return window


def stft(segment: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """Compute STFT of a segment."""
    # TODO: parametrize this
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


def samples_to_frames(interval: BubbleAnnotation, n_fft: int, hop_length: int):
    """Convert interval from sample space to stft frame space."""
    win_size_samples = interval.end - interval.start
    win_size_frames = math.floor(win_size_samples / hop_length)

    t_start = math.ceil(interval.start / hop_length)
    # t_end = math.floor(interval.end / hop_length)
    return BubbleAnnotation(start=t_start, end=t_start + win_size_frames)


class StftPreprocessor:
    """A preprocessor that applies STFT."""

    display_name: str = "Short-time Fourier transform"

    def __init__(self):
        self.n_fft = 2048
        self.hop_length = self.n_fft // 4

    def transform(self, sample):
        """Transform the sample."""
        return stft(sample, n_fft=self.n_fft, hop=self.hop_length)

    def transform_interval(self, interval: BubbleAnnotation):
        """Transform a specific interval of the sample."""
        return samples_to_frames(interval, n_fft=self.n_fft, hop_length=self.hop_length)
