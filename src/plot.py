import numpy as np


def plot_wav(ax, t, audio, title):
    """Plot the audio signal."""
    ax.plot(t, audio, linewidth=0.5, color="C0")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(0, t[-1])
    ax.set_title(title)


def plot_annotations(ax, annotations, audio, color="red", label="Annotations"):
    """Plot bubble annotations on the audio signal."""
    for i, (start_idx, end_idx) in enumerate(annotations, start=1):
        start_s = start_idx / 44100.0
        end_s = end_idx / 44100.0

        ax.axvspan(start_s, end_s, color=color, alpha=0.25, label=label if i == 1 else "")

        y_loc = 0.9 * (np.max(audio) if np.max(np.abs(audio)) > 0 else 1.0)
        ax.text((start_s + end_s) / 2, y_loc, str(i), ha="center", va="top", color="black")
