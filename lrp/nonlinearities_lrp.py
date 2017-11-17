import lrp.forward_lrp


def nonlinearities(router, R):
    """
    Handeling all nonlinearities (Relu, Sigmoid, Tanh) by passing the relevance along
    :param router: the router object to report changes to
    :param R: the list of tensors containing the relevances from the upper layers
    """
    forward_lrp.forward(router, R)
