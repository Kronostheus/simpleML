import numpy as np


class ID3Tree:
    def __init__(self: "ID3Tree") -> None:
        self.node: int = None
        self.decision: int = None
        self.branches: dict[int, ID3Tree] = None

    @staticmethod
    def get_probability(data: np.ndarray) -> np.ndarray:
        """
        Get probability (relative frequency) of every unique item in data by dividing its amount of each item
        by the size of the data
        :param data: list from which to compute probabilities
        :return: relative frequency for each unique item in data
        """
        cnt_t: np.ndarray
        _, cnt_t = np.unique(data, return_counts=True)
        return np.divide(cnt_t, len(data))

    def entropy(self: "ID3Tree", data: np.ndarray) -> float:
        """
        Compute entropy: H(data) = sum(-P(x) * log2(P(x)) for x in data)
        :param data: list from which to compute its entropy
        :return: entropy (float)
        """
        probs: np.ndarray = self.get_probability(data)
        return np.sum(np.multiply(-probs, np.log2(probs)))

    def conditional_entropy(self: "ID3Tree", feature: np.ndarray, targets: np.ndarray) -> float:
        """
        Conditional entropy quantifies the amount of information needed to describe the targets given we know the values
        of a feature. This will be used to compute the expected Information Gain of picking a given feature.

        The result of this method will be the weighted arithmetic mean of the entropies for each value of feature X:
         sum(P(x_v) * H(x_v) for x_v in X)
        :param feature: condition
        :param targets: targets (T) of data
        :return: H(T|X)
        """
        probs: np.ndarray = self.get_probability(feature)

        """
        np.take grabs the targets corresponding to the indices returned by np.nonzero
        np.flatnonzero is equivalent to a flatten np.where when only 'condition' is provided
        we compute the entropy for each possible value of feature
        """
        feat_entropy: list[float] = [self.entropy(np.take(targets, np.flatnonzero(feature == unq)))
                        for unq in np.unique(feature)]

        return np.sum(np.multiply(probs, feat_entropy))

    def split(self: "ID3Tree", features: np.ndarray, targets: np.ndarray) -> None:
        """
        Apply ID3 decision tree algorithm
        :param features:
        :param targets:
        :return:
        """

        # Check if it is pure decision (targets are all the same)
        unq_t: np.ndarray = np.unique(targets)
        if len(unq_t) == 1:
            self.decision = unq_t[0]
            return

        # Entropy of entire split
        data_entropy: float = self.entropy(targets)

        # Information Gain: IG(T, X) = H(T) - H(T|X)
        information_gain: np.ndarray = data_entropy - np.array([self.conditional_entropy(feature, targets)
                                                    for feature in features.T])

        # Get the feature that maximizes IG
        self.node: int = np.argmax(information_gain)

        # For every unique value of that feature, select its row indices
        split_indices: list[np.ndarray] = [
            np.flatnonzero(features[:, self.node] == unq)
            for unq in np.unique(features[:, self.node])
        ]

        split_idx: int
        split: np.ndarray

        # Branch out
        for split_idx, split in enumerate(split_indices):
            new_branch: ID3Tree = ID3Tree()
            new_branch.split(features[split, :], np.take(targets, split))
            self.branches[split_idx] = new_branch

    def predict(self: "ID3Tree", to_predict: np.ndarray) -> list:
        """
        Given a list of unlabeled data, traverse the tree to get targets
        :param to_predict:
        :return:
        """
        decisions: list[int] = []
        for sample in to_predict:
            curr_branch: ID3Tree = self
            while True:
                # Is leaf node (no branches)?
                if not curr_branch.branches:
                    decisions.append(curr_branch.decision)
                    break
                # Get next branch based on node feature
                curr_branch = curr_branch.branches[sample[curr_branch.node]]
        return decisions


if __name__ == "__main__":


    X: np.ndarray = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
        [2, 1, 0, 0],
        [2, 2, 1, 0],
        [2, 2, 1, 1],
        [1, 2, 1, 1],
        [0, 1, 0, 0],
        [0, 2, 1, 0],
        [2, 1, 1, 0],
        [0, 1, 1, 1],
        [1, 1, 0, 1],
        [1, 0, 1, 0],
        [2, 1, 0, 1]])

    y: np.ndarray = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0])

    tree: ID3Tree = ID3Tree()
    tree.split(X, y)
    preds: list[int] = tree.predict([
        [0, 2, 0, 1],   # Sunny, Cool, High, Strong
        [0, 2, 1, 0],   # Sunny, Cool, Normal, Weak
        [1, 1, 1, 1],   # Overcast, Mild, Normal, Strong
        [2, 2, 0, 0],   # Rainy, Cool, High, Weak
        [2, 0, 1, 1]    # Rainy, Hot, Normal, Strong
    ])
    if preds != [0, 1, 1, 1, 0]:
        raise Exception("Test failed")
