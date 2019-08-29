import numpy as np
from matplotlib import pyplot as plt

FRAME_LENGTH = 10

def prepareData(frames):
    return np.random.randint(20, size=frames)

def scatter(X, y):
    plt.scatter(X, y)
    plt.xlabel("Features")
    plt.ylabel("Mappings")
    plt.show()

def plot_fit_line(X, y, y_pred):
    plt.plot(X, y_pred, c="r")
    plt.scatter(X, y, c="b")
    plt.xlabel("Features")
    plt.ylabel("Mapping")
    plt.show()

# X = prepareData(FRAME_LENGTH)
# y = prepareData(FRAME_LENGTH)
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

scatter(X, y)

import numpy as np
from matplotlib import pyplot as plt

FRAME_LENGTH = 10

def prepareData(frames):
    return np.random.randint(20, size=frames)

def scatter(X, y):
    plt.scatter(X, y)
    plt.xlabel("Features")
    plt.ylabel("Mappings")
    plt.show()

def plot_fit_line(X, y, y_pred):
    plt.plot(X, y_pred, c="r")
    plt.scatter(X, y, c="b")
    plt.xlabel("Features")
    plt.ylabel("Mapping")
    plt.show()

X = prepareData(FRAME_LENGTH)
y = 2 + 3*X + np.random.randint(20, size=FRAME_LENGTH)

scatter(X, y)
