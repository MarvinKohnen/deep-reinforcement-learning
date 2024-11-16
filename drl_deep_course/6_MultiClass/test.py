import numpy as np


def manual_one_hot_encoding(y):
    unique_classes = np.unique(y)

    class_to_index = {cls: idx for idx, cls in enumerate(unique_classes)}

    one_hot_matrix = np.zeros((y.shape[0], len(unique_classes)))

    for i, value in enumerate(y):
        index = class_to_index[value]
        one_hot_matrix[i, index] = 1

    return one_hot_matrix


# Beispiel
y = np.array([0, 1, 2, 1, 0])
y_one_hot = manual_one_hot_encoding(y)
print(y_one_hot)
