# do not change the code in the block below
# __________start of block__________
class DummyMatch:
    def __init__(self, queryIdx, trainIdx, distance):
        self.queryIdx = queryIdx  # index in des1
        self.trainIdx = trainIdx  # index in des2
        self.distance = distance


# __________end of block__________

import numpy as np


def l2_distance_matrix(des1: np.ndarray, des2: np.ndarray) -> np.ndarray:
    """
    Args:
        des1 (np.ndarray): Descriptors from image 1, shape (N1, D)
        des2 (np.ndarray): Descriptors from image 2, shape (N2, D)

    Returns:
        np.ndarray: pairwise distances between des1 and des2 points, shape (N1, N2)

    Compute pairwise L2 distance between two descriptor matrices.

    Вычисляется сумма квадратов элементов,
    которая потом будет использоваться для вычсления расстояния между точек.
    Применяется стандартная формула линейной алгебры нормы разности векторов.
    Так как мы имеем 2 матрицы дескрипторов по 2м фотографиям,
    то каждая строчка соостветсвует вектор дескриптора (длина вектора 128).
    В конечном итоге получается матрица А, элементами которой являются расстояния между векторами.
    a_ij - расстояние между i-вектором 1ой матрицы и j-ым вектором 2ой матрицы.
    """
    d1_sq = np.sum(des1**2, axis=1, keepdims=True)
    d2_sq = np.sum(des2**2, axis=1, keepdims=True).T
    return np.sqrt(d1_sq + d2_sq - 2 * np.dot(des1, des2.T))


def match_key_points_numpy(des1: np.ndarray, des2: np.ndarray) -> list:
    """
    Match descriptors using brute-force matching with cross-check.

    Args:
        des1 (np.ndarray): Descriptors from image 1, shape (N1, D)
        des2 (np.ndarray): Descriptors from image 2, shape (N2, D)

    Returns:
        List[DummyMatch]: Sorted list of mutual best matches.
    """
    # Получаем матрицу расстояний между векторами в матрицах дескрипторов.
    distances = l2_distance_matrix(des1, des2)  # shape (N1, N2)

    # For each descriptor in des1, find best match in des2
    best_in_des2 = np.argmin(distances, axis=1)  # shape (N1,)
    # For each descriptor in des2, find best match in des1
    best_in_des1 = np.argmin(distances, axis=0)  # shape (N2,)

    # ведем поиск соответсвий по индексам.
    matches = []
    for i, j in enumerate(best_in_des2):
        if best_in_des1[j] == i:  # cross-check passed
            distance = distances[i, j]
            matches.append(DummyMatch(queryIdx=i, trainIdx=j, distance=distance))

    # Sort matches by distance (as OpenCV does)
    matches.sort(key=lambda m: m.distance)
    return matches
