from icdss.layers.activations import ReLU, Identity
from icdss.layers.dense import Dense
from icdss.layers.loss import MSE


def parse_layer(layer):
    """Parse string `layer`.

    Parameters
    ----------
    layer: str
        Layer type

    Returns
    -------
    class: icdss.layers._Layer
        Layer class
    """
    if layer.lower() == 'dense':
        return Dense
    else:
        raise ValueError('Unrecognised `layer` value %s' % layer)


def parse_activation(activation):
    """Parse string `activation`.

    Parameters
    ----------
    activation: str
        Activation type

    Returns
    -------
    class: icdss.layers._Layer
        Activation class
    """
    if activation.lower() == 'relu':
        return ReLU
    elif activation.lower() == 'identity':
        return Identity
    else:
        raise ValueError('Unrecognised `activation` value %s' % activation)


def parse_loss(loss):
    """Parse string `loss`.

    Parameters
    ----------
    loss: str
        Loss type

    Returns
    -------
    func: function
        Loss function
    """
    if loss.lower() == 'mse':
        return MSE
    else:
        raise ValueError('Unrecognised `loss` value %s' % loss)
