import numpy as np
from scipy.sparse import csr_matrix


def generate_cross_matrix(grid_shape, convolution):
    cluster_count = np.prod(grid_shape)
    cross_matrix = np.zeros((*grid_shape, *grid_shape))

    x_offset = int(np.floor(convolution.shape[0] / 2))
    y_offset = int(np.floor(convolution.shape[1] / 2))

    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            for k in range(convolution.shape[0]):
                for l in range(convolution.shape[1]):
                    cross_matrix[(
                        i, j,
                        np.clip(i + k - x_offset, 0, grid_shape[0] - 1),
                        np.clip(j + l - y_offset, 0, grid_shape[1] - 1),
                    )] += convolution[k, l]

    return csr_matrix(cross_matrix.reshape(cluster_count, cluster_count))


def generate_convolution_pattern(center, neighbour, diag, neighbour_2):
    return np.array([
        [0, 0, neighbour_2, 0, 0],
        [0, diag, neighbour, diag, 0],
        [neighbour_2, neighbour, center, neighbour, neighbour_2],
        [0, diag, neighbour, diag, 0],
        [0, 0, neighbour_2, 0, 0],
    ])
