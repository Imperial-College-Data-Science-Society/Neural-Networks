import numpy as np

from icdss.layers.base import _Layer, Linear


class Dense(_Layer):

    @staticmethod
    def _forward(x, w, b, phi):
        """`Linear` folowed by `Activation` Layers Forward Pass.

        $f(x) = \phi(w * x + b)$, with:

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
        phi: icdss.layers.Activation
            Activation function

        Returns
        -------
        out: numpy.ndaray - shape=(N, M)
            Layer output
        cache: tuple
            * linear_cache: tuple
                * x: numpy.ndarray - shape=(N, d_1, ..., d_k)
                     Input data
                * w: numpy.ndarray - shape=(D, M)
                     Layer weights
                * b: numpy.ndarray - shape=(M,)
                     Layer biases
            * phi_cache: tuple
                * a: numpy.ndarray - shape=(N, M)
                     Input data
        """
        # Forward pass from Linear Layer
        a, linear_cache = Linear._forward(x, w, b)
        # Forward pass from `Activation` Layer
        out, phi_cache = phi._forward(a)
        # Store forward passes parameters
        cache = linear_cache, phi_cache

        return out, cache

    @staticmethod
    def _backward(dout, cache, phi):
        """`Linear` folowed by `Activation` Layers Backward Pass.

        Parameters
        ----------
        dout: numpy.ndarray - any shape
            Upstream derivative
        cache: tuple
            * linear_cache: tuple
                * x: numpy.ndarray - shape=(N, d_1, ..., d_k)
                     Input data
                * w: numpy.ndarray - shape=(D, M)
                     Layer weights
                * b: numpy.ndarray - shape=(M,)
                     Layer biases
            * phi_cache: tuple
                * a: numpy.ndarray - shape=(N, M)
                     Input data
        phi: icdss.layers.Activation
            Activation function

        Returns
        ------
        grads: tuple
            * dx: numpy.ndarray - shape=(N, d_1, ..., d_k)
                 Gradient with respect to `x`
            * dw: numpy.ndarray - shape=(D, M)
                 Gradient with respect to `w`
            * db: numpy.ndarray - shape=(M,)
                 Gradient with respect to `b`
        """
        # Unwrap caches
        linear_cache, phi_cache = cache
        # Backward pass `Activation`
        da = phi._backward(dout, phi_cache)
        # Backward pass Linear
        dx, dw, db = Linear._backward(da, linear_cache)
        # Store gradients to tuple
        grads = dx, dw, db

        return grads
