import numpy as np
from utils.batch_dimension import batch_dimension


@batch_dimension(num_element_dims=2, batch_axis=-1)
def resample_edges(adjacency_matrices):
    """Randomly resamples the edges of the adjacency matrix.

    adjacency_matrices: the adjacency matrix(es). Either of shape (num_vertices, num_vertices) or (num_vertices, num_vertices, num_matrices)
    """

    resampled_matrices = np.zeros_like(adjacency_matrices)

    for index in adjacency_matrices.shape[-1]:
        matrix = adjacency_matrices[:, :, index]
        # Randomly sample a new weight for each edge
        new_weights = np.random.choice(matrix.flatten(), size=matrix.size)

        # Reshape the 1D array back into an nxn matrix
        new_adj_matrix = np.reshape(new_weights, matrix.shape)

        # Make the matrix symmetric
        new_adj_matrix = np.triu(new_adj_matrix) + np.triu(new_adj_matrix, k=1).T

        # add it to our array of resampled matrices
        resampled_matrices[:, :, index] = new_adj_matrix

    # return with the same dimensions it was passed in with
    return resampled_matrices
