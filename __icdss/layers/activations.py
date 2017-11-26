from icdss.layers.base import _Layer


class ReLU(_Layer):

    @staticmethod
    def _forward(x):
        """ReLU Layer Forward Pass.

        Parameters
        ----------
        x: numpy.ndarray - any shape
            Layer input

        Returns
        ------
        out: numpy.ndarray - shape=x.shape
            Layer output
        cache: tuple
            * x: numpy.ndarray - any shape
                 Input data
        """
        # Rectify non-positive values
        out = x * (x > 0)
        # Store forward pass parameters
        cache = x

        return out, cache

    @staticmethod
    def _backward(dout, cache):
        """ReLU Layer Backward Pass.

        Parameters
        ----------
        dout: numpy.ndarray - any shape
            Upstream derivative
        cache: tuple
            * x: numpy.ndarray - shape=dout.shape

        Returns
        -------
        dx: numpy.ndarray - shape=shape.x
            Gradient with respect to `x`
        """
        # Unwrap cache
        x = cache
        # Calculate the gradients
        dx = dout * (x > 0)

        return dx


class Identity(_Layer):

    @staticmethod
    def _forward(x):
        """Identity Layer Forward Pass.

        Parameters
        ----------
        x: numpy.ndarray - any shape
            Layer input

        Returns
        ------
        out: numpy.ndarray - shape=x.shape
            Layer output
        cache: tuple
            * x: numpy.ndarray - any shape
                 Input data
        """
        # Pass input value
        out = x
        # Store forward pass parameters
        cache = x

        return out, cache

    @staticmethod
    def _backward(dout, cache):
        """Identity Layer Backward Pass.

        Parameters
        ----------
        dout: numpy.ndarray - any shape
            Upstream derivative
        cache: tuple
            * x: numpy.ndarray - shape=dout.shape

        Returns
        -------
        dx: numpy.ndarray - shape=shape.x
            Gradient with respect to `x`
        """
        # Calculate the gradients
        dx = dout

        return dx
