
def minkowski_distance(v1, v2, p=2):
    assert len(v1) == len(v2)
    assert p != 0
    return sum([(v1i - v2i) ** p for v1i, v2i in zip(v1, v2)]) ** (1/p)

