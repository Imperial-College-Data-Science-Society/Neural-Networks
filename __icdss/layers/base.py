import numpy as np


class _Layer(object):

    @staticmethod
    def _forward(x, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _backward(dout, cache):
        raise NotImplementedError


class Linear(object):

    @staticmethod
    def _forward(x, w, b):
        """Linear Layer Forward Pass.

        $f(x) = w * x + b$, with:

        Notes
        -----
        N: int
            size of minibatch
        x[i]: numpy.ndarray - shape=(d_1, ..., d_k)
            $i_{th}$ element of minibatch 
        D: int
            d_1 * ... * d_k
        M: int
            size of output vector

        Parameters
        ----------
        x: numpy.ndarray - shape=(N, d_1, ..., d_k)
            Input data
        w: numpy.ndarray - shape=(D, M)
            Layer weights
        b: numpy.ndarray - shape=(M,)
            Layer biases

        Returns
        -------
        out: numpy.ndaray - shape=(N, M)
            Layer output
        cache: tuple
            * x: numpy.ndarray - shape=(N, d_1, ..., d_k)
                 Input data
            * w: numpy.ndarray - shape=(D, M)
                 Layer weights
            * b: numpy.ndarray - shape=(M,)
                 Layer biases
        """
        # Get number of items
        N = x.shape[0]
        # Reshape to have N rows and as many columns as needed
        x_temp = x.reshape(N, -1)
        # Simple linear operation, after this step just cache the inputs
        out = x_temp.dot(w) + b
        # Store forward pass parameters
        cache = (x, w, b)

        return out, cache

    @staticmethod
    def _backward(dout, cache):
        """Linear Layer Backward Pass.

        Notes
        -----
        N: int
            size of minibatch
        x[i]: numpy.ndarray - shape=(d_1, ..., d_k)
            $i_{th}$ element of minibatch 
        D: int
            d_1 * ... * d_k
        M: int
            size of output vector

        Parameters
        ----------
        dout: numpy.ndarray - shape=(N, M)
            Upstream derivative
        cache: tuple
            * x: numpy.ndarray - shape=(N, d_1, ..., d_k)
                 Input data
            * w: numpy.ndarray - shape=(D, M)
                 Layer weights
            * b: numpy.ndarray - shape=(M,)
                 Layer biases

        Returns
        -------
        grads: tuple
            * dx: numpy.ndarray - shape=(N, d_1, ..., d_k)
                 Gradient with respect to `x`
            * dw: numpy.ndarray - shape=(D, M)
                 Gradient with respect to `w`
            * db: numpy.ndarray - shape=(M,)
                 Gradient with respect to `b`
        """
        # Unwrap cache
        x, w, b = cache
        # Get number of items
        N = x.shape[0]
        # Reshape to have N rows and as many collumns as needed
        x_temp = x.reshape(N, -1)
        # Calculate the gradients; for reference check http://cs231n.github.io/optimization-2/#mat
        db = np.sum(dout, axis=0)
        dw = np.dot(x_temp.T, dout)
        dx = np.dot(dout, w.T).reshape(x.shape)
        # Store gradients to tuple
        grads = dx, dw, db

        return grads
