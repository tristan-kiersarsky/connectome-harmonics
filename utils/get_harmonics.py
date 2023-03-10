import numpy as np
from utils.batch_dimension import batch_dimension
import pygsp


@batch_dimension(num_element_dims=2, batch_axis=-1)
def get_harmonics(adjacency_matrices, lap_type="normalized"):
    """Returns the eigenvalues and eigenvectors of the graph laplacian for the provided adjacency matrices.

    adjacency_matrices: the adjacency matrix(es). Either of shape (num_vertices, num_vertices) or (num_vertices, num_vertices, num_matrices)
    lap_type: either "combinatorial" or "normalized"
    """

    num_vertices, _, num_matrices = adjacency_matrices.shape
    harmonics = np.zeros_like(adjacency_matrices)
    eigval = np.zeros((num_vertices, num_matrices))
    for i in range(num_matrices):
        adjacency_matrix = adjacency_matrices[:, :, i]
        np.fill_diagonal(adjacency_matrix, 0)  # PyGSP does not support self-loops
        G_fd = pygsp.graphs.Graph(adjacency_matrix)  # PyGSP graph
        G_fd.compute_laplacian(lap_type=lap_type)
        G_fd.compute_fourier_basis()  # compute connectome harmonics

        harmonics[:, :, i] = G_fd.U
        eigval[:, i] = G_fd.e

    return harmonics, eigval
