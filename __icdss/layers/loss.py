import numpy as np


def MSE(y_hat, y):
    """Mean Squared Error Loss Function.

    $MSE(y, yhat) = \frac{1}{2N}\sum_{i=1}^{N}(y_{i} - \hat_{y}_{i})^{2}$

    Parameters
    ----------
    y: numpy.ndarray - any shape
        True targets
    y_hat: numpy.ndarray - shape=y.shape
        Predicted outputs

    Returns
    -------
    loss: float
        Loss value
    dy: numpy.ndarray - shape=y.shape
        Gradient of loss with respect to y
    """
    # Store gradient vector
    dy = y_hat - y
    # MSE calculation
    loss = (dy ** 2).mean(axis=1)

    return loss, dy
