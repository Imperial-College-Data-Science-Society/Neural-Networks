# scientific computing library
import numpy as np

# visualization tools
import matplotlib.pyplot as plt


def linear(X, y, W, b, h=0.02, o=0.2):
    """Visualize Separation Lines - Linear Model.

    Parameters
    ----------
    X: numpy.ndarray
        Features matrix
    y: numpy.ndarray
        Targets/Labels matrix
    W: numpy.ndarray
        Weights matrix
    b: numpy.ndarray
        Biases vector
    h: float
        Plot density
    o: float
        Plot axis offset
    """

    # Plot ranges
    x_min, x_max = X[:, 0].min() - o, X[:, 0].max() + o  # x-range
    y_min, y_max = X[:, 1].min() - o, X[:, 1].max() + o  # y-range

    # 2D grid values of density `h`
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    # Classify points
    Z = np.argmax(np.dot(np.c_[xx.ravel(), yy.ravel()],
                         W) + b, axis=1).reshape(xx.shape)

    # Plot separation spaces
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    # Plot input datapoints
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.title('Linear Model for Spiral Toy Data\n')
    plt.xlabel('$x_{1}$')
    plt.ylabel('$x_{2}$')


def nn(X, y, W1, b1, W2, b2, h=0.02, o=0.2):
    """Visualize Separation Lines - Neural Network.

    Parameters
    ----------
    X: numpy.ndarray
        Features matrix
    y: numpy.ndarray
        Targets/Labels matrix
    W1: numpy.ndarray
        Weights matrix of first layer
    b1: numpy.ndarray
        Biases vector of first layer
    W2: numpy.ndarray
        Weights matrix of second layer
    b2: numpy.ndarray
        Biases vector of second layer
    h: float
        Plot density
    o: float
        Plot axis offset
    """

    # ReLU activation function
    def relu(z): return np.maximum(0, z)

    # Plot ranges
    x_min, x_max = X[:, 0].min() - o, X[:, 0].max() + o  # x-range
    y_min, y_max = X[:, 1].min() - o, X[:, 1].max() + o  # y-range

    # 2D grid values of density `h`
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    # Classify points
    Z = np.argmax(np.dot(relu(np.dot(
        np.c_[xx.ravel(), yy.ravel()], W1) + b1), W2) + b2, axis=1).reshape(xx.shape)

    # Plot separation spaces
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    # Plot input datapoints
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.title('Neural Network Model for Spiral Toy Data\n')
    plt.xlabel('$x_{1}$')
    plt.ylabel('$x_{2}$')
