"""The 'is this render blank?' checker, tested on images small enough to count by hand."""

import numpy as np
import pytest
from support.images import foreground_fraction


def image(height, width, color=(0, 0, 0)):
    return np.tile(np.array(color, dtype=np.uint8), (height, width, 1))


def test_a_uniform_image_has_no_foreground():
    assert foreground_fraction(image(10, 10, (7, 7, 7))) == 0.0


def test_the_left_half_in_another_color_is_half_the_pixels():
    picture = image(10, 10)
    picture[:, :5] = (200, 30, 30)
    assert foreground_fraction(picture) == pytest.approx(0.5)


def test_a_single_different_pixel_is_one_percent_of_a_ten_by_ten_image():
    picture = image(10, 10)
    picture[3, 4] = (255, 255, 255)
    assert foreground_fraction(picture) == pytest.approx(0.01)


def test_small_color_noise_is_not_foreground():
    noise = np.random.default_rng(0).integers(-5, 6, size=(10, 10, 3))
    picture = np.clip(100 + noise, 0, 255).astype(np.uint8)
    assert foreground_fraction(picture) == 0.0


@pytest.mark.parametrize(("difference", "counts"), [(31, True), (30, False), (29, False)])
def test_a_difference_counts_only_when_it_is_more_than_the_threshold(difference, counts):
    picture = image(10, 10)
    picture[0, 0] = (difference, 0, 0)
    assert (foreground_fraction(picture) > 0) is counts


def test_an_image_that_is_not_square_is_measured_over_all_its_pixels():
    picture = image(4, 10)
    picture[0, :] = (255, 0, 0)
    assert foreground_fraction(picture) == pytest.approx(0.25)


@pytest.mark.parametrize(("channels", "counts"), [((11, 11, 11), True), ((9, 9, 9), False)])
def test_differences_in_several_channels_add_up(channels, counts):
    picture = image(10, 10)
    picture[0, 0] = channels
    assert (foreground_fraction(picture) > 0) is counts
