class CudaExecutionPipeline:
    def __init__(self, backend, layers: List[CudaMoELayer], disk_loader):
        self.backend = backend
        self.layers = layers
        self.disk_loader = disk_loader

    def step(self, hidden_states: torch.Tensor, current_layer_idx: int):
        lookahead_layer_idx = current_layer_idx + 1

        # 1. Prerouter: Predict next layer's experts from current activations
        if lookahead_layer_idx < len(self.layers):
            predicted_experts = self.layers[lookahead_layer_idx].predict_experts(hidden_states)
            
            # 2. Trigger asynchronous PCIe transfers for next layer in transfer_stream
            for exp_id in predicted_experts:
                host_weights = self.disk_loader.fetch_pinned(exp_id)
                event = self.backend.expert_cache.prefetch_expert_async(
                    expert_id=exp_id,
                    host_expert_tensor=host_weights,
                    transfer_stream=self.backend.transfer_stream
                )
                self.backend.transfer_ready_events[exp_id] = event

        # 3. Compute current layer in compute_stream
        output = self.layers[current_layer_idx](hidden_states)
        return output