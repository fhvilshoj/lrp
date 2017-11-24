from lrp import forward_lrp


def printing(router, R):
    """
    Handeling all nonlinearities (Relu, Sigmoid, Tanh) by passing the relevance along
    :param router: the router object to report changes to
    :param R: the list of tensors containing the relevances from the upper layers
    """
    # Sum the potentially multiple relevances from the upper layers
    forward_lrp.forward(router, R)
