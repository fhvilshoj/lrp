def nonlinearities(path, R):
    """
    Handeling all nonlinearities (Relu, Sigmoid, Tanh) by passing the relevance along
    :param path: the path of operations towards the input
    :param R: The upper layer relevance
    :return: lower layer relevance
    """

    assert path[0].outputs[0].shape == R.shape, "Tensor and R should have same shape"

    # Skip this operation and pass R along with the next operation in the path
    return path[1:], R


