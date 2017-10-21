import lrp_util

def softmax(router, R):
    """
    Handeling softmax layers by passing the relevance along to the input
    :param router: the router object to report changes to
    :param R: the list of tensors containing the relevances from the upper layers
    """
    # Sum the potentially multiple relevances from the upper layers
    R = lrp_util.sum_relevances(R)

    # Get the current operation
    current_operation = router.get_current_operation()

    # Report handled operations
    router.mark_operation_handled(current_operation)

    # Forward the calculated relevance to the input of the convolution
    router.forward_relevance_to_operation(R, current_operation, current_operation.inputs[0].op)
