from scipy.stats import entropy
from utils.batch_dimension import batch_dimension
import numpy as np


def get_entropy(a, binning_frequency=100):
    """Computes the bits of entropy of the distribution of values in the eigenvector,
    after putting the values into uniformly sized bins.

    args:
        a: array to compute entropy over.
        binning_frequency: how many bins per unit. i.e. 100 means there will be 100 bins
        axis: the axis to operate on. If None, the array will be flattened.
    """
    binned_vals = np.round(a * binning_frequency).astype(int)
    values, counts = np.unique(binned_vals, return_counts=True)
    return entropy(counts, base=2)


@batch_dimension(num_element_dims=2, batch_axis=-1)
def get_harmonics_entropy(harmonics, binning_frequency=100, num_harmonics=200):
    """Computes the entropy of the first n harmonics.

    args:
    harmonics: shape (harmonic_dimension, harmonic_dimension, num_subjects) array of neural harmonics for each subject
    binning_frequency: see help for get_entropy()
    num_harmonics: calculate entropy for the first n harmonics lowest-frequency harmonics.
    """
    harmonic_dim, n, num_subjects = harmonics.shape
    assert (
        n >= num_harmonics
    ), f"requested first {num_harmonics}, but there are only {n} harmonics."

    harmonic_entropy = np.zeros((num_subjects, num_harmonics))

    for subject_idx in range(num_subjects):
        for harmonic_idx in range(num_harmonics):
            harmonic = harmonics[:, harmonic_idx, subject_idx]
            harmonic_entropy[subject_idx, harmonic_idx] = get_entropy(
                harmonic, binning_frequency=binning_frequency
            )

    return harmonic_entropy
