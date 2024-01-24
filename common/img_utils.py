import numpy as np

def calculate_mean_and_variance(all_images):
    mean = np.mean(all_images, axis=(0, 1, 2))
    variance = np.var(all_images, axis=(0, 1, 2))

    return mean, variance