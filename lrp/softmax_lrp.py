import lrp.forward_lrp


def softmax(router, R):
    """
    Handeling softmax layers by passing the relevance along to the input
    :param router: the router object to report changes to
    :param R: the list of tensors containing the relevances from the upper layers
    """
    forward_lrp.forward(router, R)
