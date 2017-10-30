from context_handler import ContextHandler
from lrp import lrp_util
from lrp.convolutional_lrp import convolutional
from lrp.linear_lrp import linear, elementwise_linear, sparse_dense_linear
from lrp.max_pooling_lrp import max_pooling
from lrp.nonlinearities_lrp import nonlinearities
from lrp.softmax_lrp import softmax
from lrp.shaping_lrp import shaping, sparse_shaping
from lrp.concatenate_lrp import concatenate
from lrp.split_lrp import split
from tile_lrp import tile

from constants import *
from print_lrp import printing
from slicing_lrp import slicing


class StandardContextHandler(ContextHandler):
    # Known operations to the standard context handler
    _router = {
        'MatMul': linear,
        'Mul': elementwise_linear,
        'SparseTensorDenseMatMul': sparse_dense_linear,
        'Conv2D': convolutional,
        'MaxPool': max_pooling,
        'ExpandDims': shaping,
        'Squeeze': shaping,
        'Reshape': shaping,
        'SparseReshape': sparse_shaping,
        'ConcatV2': concatenate,
        'Split': split,
        'Relu': nonlinearities,
        'Sigmoid': nonlinearities,
        'Tanh': nonlinearities,
        'Softmax': softmax,
        'Slice': slicing,
        'Print': printing,
        'Tile': tile
    }

    def __init__(self, router):
        super().__init__(router)
        self.current_path_index = 0

    def get_current_operation(self):
        # return the operation at the current path index of the context path
        return self.context[CONTEXT_PATH][self.current_path_index]

    def handle_context(self, context):
        # Reset current path index if the handler was used before
        self.current_path_index = 0
        # Remember the new context
        self.context = context
        # Extract the current path
        current_path = context[CONTEXT_PATH]

        # Run through the operations in the path
        while self.current_path_index < len(current_path):
            current_operation = current_path[self.current_path_index]

            # If the operation has already been taken care of, skip it
            # by jumping to next while iteration
            if self.router.is_operation_handled(current_operation):
                self.current_path_index += 1
                continue

            # Find type of the operation in the front of the path
            operation_type = current_operation.type
            if operation_type in ['Add', 'BiasAdd']:
                # Check which operation a given addition is associated with
                # Note that it cannot be lstm because lstm has its own scope
                operation_type = lrp_util.addition_associated_with(current_operation.outputs[0])

            if operation_type in self._router:
                # Route responsibility to appropriate function
                # Send the recorded relevance for the current operation
                # along. This saves the confusion of finding relevances
                # for Add in the concrete implementations
                self._router[operation_type](self, self.router.get_relevance_for_operation(current_operation))
            else:
                print("Router did not know the operation: ", operation_type)
                # If we don't know the operation, skip it

            # Go to next operation in path
            self.current_path_index += 1
