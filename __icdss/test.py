import numpy as np
import matplotlib.pyplot as plt

import icdss

net = icdss.nn.Network([1, 5, 10, 1])

X = np.linspace(0, 1, 100).reshape(-1, 1)
y = (7 * X + 1).reshape(-1, 1)

epochs = 20

hist = []

for i in range(epochs):
    _loss = 0
    for xi, yi in zip(X, y):
        loss, grads = net._step(xi, yi)
        _loss += loss
    hist.append(_loss / X.shape[0])
    net._update(grads, 0.01)

y_hat = net.predict(X)

plt.plot(y, label='Target')
plt.plot(y_hat, label='Model')
plt.legend()
plt.show()
