class ContextHandler:
    def __init__(self, router):
        self.router = router
        self.context = None

    # Forward to real router
    def forward_relevance_to_operation(self, r, ir, oR):
        self.router.forward_relevance_to_operation(r, ir, oR)

    # Forward to real router
    def mark_operation_handled(self, operation):
        self.router.mark_operation_handled(operation)

    # Forward to real router
    def did_add_extra_dimension_for_multiple_predictions_per_sample(self):
        return self.router.added_dimension_for_multiple_predictions_per_sample

    def get_current_operation(self):
        pass

    def handle_context(self, context):
        pass
