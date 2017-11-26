import numpy as np
# Parsers for network configuration parameters
from icdss.layers import parse_activation
from icdss.layers import parse_layer
from icdss.layers import parse_loss
# Used for the last layer in regression mode
from icdss.layers import Identity


class Network(object):

    def __init__(self, n_dims, layer='dense', activation='relu', loss='mse', mode='regression'):
        """Initialize Neural Network.

        Parameters
        ----------
        n_dims: list
            Dimensions configuration of the network
        layer: str
            Hidden layers structure {'dense'}
        activation: str
            Activation function {'relu'}
        loss: str
            Loss function {'mse'}
        mode: str
            Regression vs Classification network {'regression', 'classification'}
        """
        # Parse `layer` string
        self.layer = parse_layer(layer)
        # Parse `activation` string
        self.activation = parse_activation(activation)
        # Parse `loss` string
        self.loss = parse_loss(loss)
        # Network mode (classification or regression)
        self.mode = self._parse_mode(mode)

        # Number of layers, input layer does not count
        self.num_layers = len(n_dims) - 1
        # Network parameters
        self.params = {}

        # initialize parameters randomly
        for j, shape in enumerate(zip(n_dims[:-1], n_dims[1:])):
            self.params['W' + str(j + 1)] = np.random.normal(0, 0.01, shape)
            self.params['b' + str(j + 1)] = np.zeros(shape[1])

    def _forward(self, X):
        """Forward pass from the `Network`.

        Parameters
        ----------
        X: numpy.ndarray
            Input (batch) data

        Returns
        -------
        y_hat: numpy.ndarray
            Predicted output
        cache: tuple
            * _cache:
        """
        # Keep track of input moving through network
        _input = X
        # Store parameter values for backward pass
        cache = []

        # Iterate over all hidden layers
        for j in range(1, self.num_layers):
            # Network parameters keys
            _W = 'W' + str(j)
            _b = 'b' + str(j)
            # Update `_input` and store params to `_cache`
            _input, _cache = self.layer._forward(
                _input, self.params[_W], self.params[_b], self.activation)
            cache.append(_cache)

        # Last layer pass
        _W = 'W' + str(self.num_layers)
        _b = 'b' + str(self.num_layers)
        # Last layer activation function
        phi = None
        if self.mode == 'classification':
            raise NotImplementedError
        elif self.mode == 'regression':
            phi = Identity
        y_hat, _cache_out = self.layer._forward(
            _input, self.params[_W], self.params[_b], phi)
        # Store params
        cache.append(_cache_out)

        return y_hat, cache

    def _backward(self, dout, cache):

        # Parameter gradients w.r.t loss function
        grads = {}

        # Last layer pass
        _W = 'W' + str(self.num_layers)
        _b = 'b' + str(self.num_layers)
        # Last layer activation function
        phi = None
        if self.mode == 'classification':
            raise NotImplementedError
        elif self.mode == 'regression':
            phi = Identity
        # Backward pass to last layer & store gradients
        layer_dx, grads[_W], grads[_b] = self.layer._backward(
            dout, cache.pop(), phi)

        # Iterate over all hidden layers (in REVERSE order)
        for j in range(1, self.num_layers):
            # Network parameters keys
            _W = 'W' + str(self.num_layers - j)
            _b = 'b' + str(self.num_layers - j)
            # Backward pass to hidden layers & store gradients
            layer_dx, grads[_W], grads[_b] = self.layer._backward(
                layer_dx, cache.pop(), self.activation)

        return grads

    def _step(self, X, y):
        # Forward pass
        y_hat, cache = self._forward(X)
        # Loss function
        loss, dy = self.loss(y_hat, y)
        # Backward pass
        grads = self._backward(dy, cache)

        return loss, grads

    def _update(self, grads, eta):
        """Parameters update.

        Parameters
        ----------
        grads: dict
            Gradients of network parameters
        eta: float
            Learning rate for SGD
        """
        for p, w in self.params.items():
            self.params[p] -= eta * grads[p]

    def _parse_mode(self, mode):
        if (mode != 'regression') and (mode != 'classification'):
            raise ValueError('Unrecognised `mode` parameter %s' % mode)
        else:
            return mode

    def fit(self, X, y, epochs=100, batch_size=1, eta=0.01):
        pass

    def predict(self, X):
        return self._forward(X)[0]
