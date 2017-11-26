import numpy as np


class FullyConnectedNet(object):
    """A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {Linear - ReLU} x (L - 1) - Linear

    where the {...} block is repeated L - 1 times.
    """

    def __init__(self, n_dims, reg=0.0, weight_scale=1e-2):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.

        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        """
        self.reg = reg
        self.num_layers = len(n_dims) - 1
        self.params = {}

        for i, shape in enumerate(zip(n_dims[:-1], n_dims[1:])):
            self.params['W' + str(i + 1)] = np.random.normal(0,
                                                             weight_scale, shape)
            self.params['b' + str(i + 1)] = np.zeros(shape[1])

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        mode = 'test' if y is None else 'train'

        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        current_input = X
        linear_relu_cache = {}

        for i in range(1, self.num_layers):
            _W = 'W' + str(i)
            _b = 'b' + str(i)
            current_input, linear_relu_cache[i] = Layer.forward(
                current_input, self.params[_W], self.params[_b])

        # Last linear layer:
        _W = 'W' + str(self.num_layers)
        _b = 'b' + str(self.num_layers)
        linear_out, linear_cache = Linear.forward(
            current_input, self.params[_W], self.params[_b])
        scores = linear_out
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################

        #print(scores, y, self.params['W' + str(self.num_layers)].shape)
        loss, dscores = MSE(scores, y)

        # last layer:
        linear_dx, linear_dw, linear_db = Linear.backward(
            dscores, linear_cache)
        grads['W' + str(self.num_layers)] = linear_dw + \
            self.reg * self.params['W' + str(self.num_layers)]
        grads['b' + str(self.num_layers)] = linear_db
        loss += 0.5 * self.reg * \
            np.sum(self.params['W' + str(self.num_layers)]**2)

        for i in range(self.num_layers - 1, 0, -1):

            linear_dx, linear_dw, linear_db = Layer.backward(
                linear_dx, linear_relu_cache[i])

            _W = 'W' + str(i)
            _b = 'b' + str(i)
            loss += 0.5 * self.reg * np.sum(self.params[_W]**2)
            grads[_W] = linear_dw + self.reg * self.params[_W]
            grads[_b] = linear_db
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

    def fit(self, X, y, epochs=100, batch_size=1, eta=0.01):
        # number of input datapoints
        N = X.shape[0]
        # number of batches
        it_per_epoch = max(int(N / batch_size), 1)
        num_it = epochs * it_per_epoch

        loss_hist = []

        for j in range(num_it):
            batch_mask = np.random.choice(N, batch_size)
            X_batch = X[batch_mask]
            y_batch = y[batch_mask]

            loss, grads = self.loss(X_batch, y_batch)

            # parameters update
            for p, w in self.params.items():
                self.params[p] -= eta * grads[p]

            loss_hist.append(loss)

        return loss_hist
