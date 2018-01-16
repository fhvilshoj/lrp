from lrp.configuration import LAYER, ZbConfiguration
from lrp import lrp_util
from lrp.context_handler import ContextHandler
from lrp.convolutional_lrp import convolutional
from lrp.linear_lrp import linear, elementwise_linear, sparse_dense_linear
from lrp.max_pooling_lrp import max_pooling
from lrp.nonlinearities_lrp import nonlinearities
from lrp.softmax_lrp import softmax
from lrp.shaping_lrp import shaping
from lrp.concatenate_lrp import concatenate
from lrp.split_lrp import split
from lrp.sparse_reorder_lrp import sparse_reorder
from lrp.tile_lrp import tile
from lrp.print_lrp import printing
from lrp.slicing_lrp import slicing
from lrp.sparse_reshape_lrp import sparse_reshape
from lrp.constants import *


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
        'ConcatV2': concatenate,
        'Split': split,
        'Relu': nonlinearities,
        'Sigmoid': nonlinearities,
        'Tanh': nonlinearities,
        'Softmax': softmax,
        'Slice': slicing,
        'Print': printing,
        'Tile': tile,
        'SparseReshape': sparse_reshape,
        'SparseReorder': sparse_reorder
    }

    def __init__(self, router):
        super().__init__(router)
        self.current_path_index = 0
        self.zb_info = {}
        self.zb_path_index = -1

    def get_current_operation(self):
        # return the operation at the current path index of the context path
        return self.context[CONTEXT_PATH][self.current_path_index]

    def get_configuration(self, layer):
        if self.current_path_index == self.zb_path_index:
            # Check if final layer and return zb configuration
            return ZbConfiguration(**self.zb_info)
        else:
            return super(StandardContextHandler, self).get_configuration(layer)

    def _find_add_if_exists(self, reverse_index):
        path = self.context[CONTEXT_PATH]
        path = path[:len(path) - reverse_index - 1]

        for i, op in enumerate(reversed(path)):
            if op.type in ['Add', 'BiasAdd']:
                return reverse_index + 1 + i
            elif self._is_real_layer(op):
                return reverse_index

        return reverse_index

    @staticmethod
    def _is_real_layer(op):
        return op.type in ['MatMul', 'Mul', 'Conv2D', 'MaxPool', 'Softmax']

    @staticmethod
    def _is_zb_supported(op):
        return op.type in ['MatMul', 'Conv2D']

    def _find_first_real_layer(self):
        current_path = self.context[CONTEXT_PATH]
        for rev_idx, op in enumerate(reversed(current_path)):
            if self._is_real_layer(op):
                if self._is_zb_supported(op):
                    # First real layer supports zb rule and zb rule is activated
                    self.zb_path_index = len(current_path) - self._find_add_if_exists(rev_idx) - 1
                break

    def handle_context(self, context):

        # Reset current path index if the handler was used before
        self.current_path_index = 0
        # Remember the new context
        self.context = context

        self.zb_info = self.router.get_first_layer_zb()
        if self.router.is_final_context and self.zb_info:
            self._find_first_real_layer()

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
                # If we don't know the operation, skip it
                pass

            # Go to next operation in path
            self.current_path_index += 1
