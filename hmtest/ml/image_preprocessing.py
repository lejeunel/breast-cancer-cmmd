import numpy as np
from scipy.signal import find_peaks
from skimage.measure import label, regionprops


def triangle_thresholding(image: np.ndarray, n_bins=256) -> np.ndarray:
    """
    Apply triangle thresholding and return a binary map.

    Ref: Zack, G.W.; Rogers, W.E.; Latt, S.A. Automatic measurement of sister chromatid exchange frequency. J. Histochem. Cytochem. Off.
    J. Histochem. Soc. 1977, 25, 741–753.

    """

    # Compute the histogram
    hist, bins = np.histogram(image.flatten(), n_bins, range=[0, n_bins])

    # Find the peak of the histogram
    peaks, _ = find_peaks(hist)

    # Calculate Triangle threshold
    triangle_threshold = (peaks[0] + np.argmax(hist[peaks[0] :])) // 2

    return image >= triangle_threshold


def find_largest_connected_components(binary_mask: np.ndarray):
    """
    Takes a binary mask and return another mask with
    the largest connected region
    """
    labels = label(binary_mask)

    areas = [(i, region.area) for i, region in enumerate(regionprops(labels))]
    max_region = max(areas, key=lambda l: l[1])
    max_region_label = max_region[0]

    return binary_mask == max_region_label
