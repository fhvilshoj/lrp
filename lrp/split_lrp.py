import tensorflow as tf
import lrp_util


# When we see a split we want to concatenate the incoming relevances
def split(router, R):
    # Get the current split operation
    current_operation = router.get_current_operation()

    relevances_from_ops = dict()
    for r in R:
        if not r['producer'] in relevances_from_ops:
            relevances_from_ops[r['producer']] = [r['relevance']]
        else:
            relevances_from_ops[r['producer']].append(r['relevance'])

    # Order relevance lists according to the order that they appear in
    # as input to the node that they came from
    sorted_relevances_from_ops = dict()
    for key, value in relevances_from_ops.items():
        # We only need to sort lists longer than one
        sorted_relevances = value
        if len(value) > 1:
            sorted_relevances = []
            inputs = current_operation.graph._nodes_by_id[key].inputs

            def _is_in_list(tensor):
                for v in value:
                    if v is tensor:
                        return True
                return False

            for i in inputs:
                if _is_in_list(i):
                    sorted_relevances.append(i)
        sorted_relevances_from_ops[key] = sorted_relevances

    # Find axis
    axis = current_operation.inputs[0]

    # Sum relevances for each output
    relevances_to_concatenate = []
    for output_index, output in enumerate(current_operation.outputs):
        relevance_sum = tf.zeros_like(output)
        for consumer in output.consumers():
            if consumer._id in sorted_relevances_from_ops:
                rel = sorted_relevances_from_ops[consumer._id]
                if len(rel) > 0:
                    relevance_sum += rel.pop(0)

        relevances_to_concatenate.append(relevance_sum)

    # Concatenate relevances in same order
    R_concatenated = tf.concat(relevances_to_concatenate, axis)

    # Tell the router that we handled this operation
    router.mark_operation_handled(current_operation)

    # Forward relevance to the operation of the input to the current operation
    router.forward_relevance_to_operation(R_concatenated, current_operation, current_operation.inputs[1].op)
