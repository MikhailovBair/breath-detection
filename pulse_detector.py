import cv2
import numpy as np

from scipy.fft import fft, fftfreq, rfft, rfftfreq


def pulse_from_breath(breath_frequency):
    return 0.9 + breath_frequency + 0.48


def crop_fragment(img, face):
    x, y, w, h = face
    return img[y + h // 15 : y + h // 5, x + w // 3 : x + 2 * w // 3]


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def pulse_from_sample(sample, rotate=False, fps=30):
    sample = sample[:-4] - moving_average(sample, 5)
    sample = sample / max(abs(sample))

    fft_y = rfft(sample)
    fft_x = rfftfreq(len(sample), 1 / fps)

    mask = (fft_x >= 0.8) & (fft_x <= 3.7)
    return fft_x[mask][abs(fft_y[mask]).argmax()]
