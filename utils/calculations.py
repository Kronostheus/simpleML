import numpy as np


def minkowski_distance(v1: float, v2: float, p: int = 2) -> float:
    if len(v1) != len(v2):
        raise ValueError("Vectors have different lengths")
    if p <= 0:
        raise ValueError("P should b positive")
    return sum([(v1i - v2i) ** p for v1i, v2i in zip(v1, v2)]) ** (1/p)


def levenshtein_distance(
        first_sequence: str,
        second_sequence: str,
        del_cost: int = 1,
        ins_cost: int = 1,
        sub_cost: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:

    rows: int = len(first_sequence) + 1
    cols: int = len(second_sequence) + 1

    distances: np.ndarray = np.zeros(shape=(rows, cols), dtype=int)

    for row in range(1, rows):
        distances[row, 0] = del_cost * row

    for col in range(1, cols):
        distances[0, col] = ins_cost * col

    for col in range(1, cols):
        for row in range(1, rows):
            is_substitution: int = 0 if first_sequence[row-1] == second_sequence[col-1] else sub_cost
            distances[row, col] = min(
                distances[row-1, col] + del_cost,
                distances[row, col-1] + ins_cost,
                distances[row - 1, col - 1] + is_substitution
            )

    return distances[rows-1, cols-1], distances
