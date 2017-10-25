class ContextHandler:
    def __init__(self, router):
        self.router = router
        self.context = None

    # Forward to real router
    def forward_relevance_to_operation(self, relevance, relevance_producer, relevance_receiver):
        self.router.forward_relevance_to_operation(relevance, relevance_producer, relevance_receiver)

    # Forward to real router
    def mark_operation_handled(self, operation):
        self.router.mark_operation_handled(operation)

    # Forward to real router
    def starting_point_relevances_did_not_have_predictions_per_sample_dimension(self):
        return self.router.starting_point_relevances_did_not_have_predictions_per_sample_dimension

    def get_current_operation(self):
        pass

    def handle_context(self, context):
        pass
