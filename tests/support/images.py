"""Checks on rendered images."""

import numpy as np


def foreground_fraction(image, threshold=30):
    """Fraction of pixels that differ from the most common colour by more than `threshold`.

    The most common colour is taken to be the background, which is safe for a render of one arm
    that fills a small part of the frame. The difference is summed over the colour channels.
    """
    pixels = image.reshape(-1, image.shape[-1]).astype(int)
    colors, counts = np.unique(pixels, axis=0, return_counts=True)
    background = colors[counts.argmax()]
    return float((np.abs(pixels - background).sum(axis=1) > threshold).mean())
