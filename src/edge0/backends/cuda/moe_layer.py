import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import grouped_gemm
except ImportError:
    grouped_gemm = None

class CudaMoELayer(nn.Module):
    """
    MoE Layer executing streaming expert compute and Recover-LoRA on CUDA.
    """
    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        num_experts: int,
        top_k: int,
        lora_rank: int = 16,
        backend=None
    ):
        super().__init__()
        self.backend = backend
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.lora_rank = lora_rank

        # Router Gate
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)

        # Recover-LoRA adapters (stored in GPU memory permanently)
        self.lora_A = nn.Parameter(torch.randn(num_experts, hidden_dim, lora_rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(num_experts, lora_rank, intermediate_dim))
        self.lora_scaling = 1.0 / lora_rank

    def forward(
        self,
        x: torch.Tensor,
        predicted_routing_tokens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward step coordinated across compute stream.
        """
        batch_seq, d_model = x.shape
        
        with torch.cuda.stream(self.backend.compute_stream):
            # 1. Router logits and top-k selection
            logits = self.gate(x)
            routing_weights = F.softmax(logits, dim=-1)
            scores, top_indices = torch.topk(routing_weights, self.top_k, dim=-1)
            scores = scores / scores.sum(dim=-1, keepdim=True)

            out = torch.zeros_like(x)

            # 2. Iterate through chosen active experts
            unique_experts = torch.unique(top_indices).tolist()
            for exp_id in unique_experts:
                # Synchronize compute stream with the transfer completion of exp_id
                if exp_id in self.backend.transfer_ready_events:
                    self.backend.compute_stream.wait_event(
                        self.backend.transfer_ready_events[exp_id]
                    )

                expert_weights = self.backend.expert_cache.get_expert_weights(exp_id)
                
                # Identify token masks mapped to this expert
                mask = (top_indices == exp_id).any(dim=-1)
                if not mask.any():
                    continue

                sub_x = x[mask]

                # Base expert GEMM
                base_out = torch.matmul(sub_x, expert_weights)

                # Recover-LoRA branch: sub_x @ lora_A @ lora_B
                lora_out = (
                    sub_x @ self.lora_A[exp_id]
                ) @ self.lora_B[exp_id] * self.lora_scaling

                combined = base_out + lora_out
                out[mask] += combined * scores[mask, (top_indices[mask] == exp_id).nonzero(as_tuple=True)].unsqueeze(-1)

            return out

class CUTLASSGDSMoE(nn.Module):
    """
    MoE layer combining GPUDirect Storage expert caching with
    CUTLASS Grouped GEMM execution.
    """
    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        num_experts: int,
        top_k: int,
        gds_cache,
        lora_rank: int = 16,
        device: str = "cuda:0"
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.gds_cache = gds_cache
        self.device = torch.device(device)

        # Gate router
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False, device=self.device)

        # Recover-LoRA adapters (permanently pinned in GPU VRAM)
        self.lora_A = nn.Parameter(
            torch.randn(num_experts, hidden_dim, lora_rank, device=self.device) * 0.01
        )
        self.lora_B = nn.Parameter(
            torch.zeros(num_experts, lora_rank, intermediate_dim, device=self.device)
        )
        self.lora_scaling = 1.0 / lora_rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes sorted token gathering, waits on GDS transfers, 
        and launches CUTLASS Grouped GEMM.
        """
        batch_size, seq_len, d_model = x.shape
        flat_x = x.view(-1, d_model)
        num_tokens = flat_x.shape[0]

        # 1. Routing calculation
        logits = self.gate(flat_x)
        routing_weights = F.softmax(logits, dim=-1)
        scores, top_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        scores = scores / scores.sum(dim=-1, keepdim=True)

        # Flatten routing assignments
        flat_top_indices = top_indices.view(-1)
        flat_scores = scores.view(-1, 1)

        # 2. Sort tokens by expert ID to construct contiguous variable-M blocks
        sorted_expert_indices, token_sort_order = torch.sort(flat_top_indices)
        orig_token_idx = (
            torch.arange(num_tokens, device=self.device)
            .unsqueeze(1)
            .expand(-1, self.top_k)
            .reshape(-1)[token_sort_order]
        )

        # Find active expert distribution
        unique_experts, expert_token_counts = torch.unique_consecutive(
            sorted_expert_indices, return_counts=True
        )
        active_expert_list = unique_experts.tolist()
        num_active = len(active_expert_list)

        # Pack input activations contiguously for CUTLASS
        gathered_x = flat_x[orig_token_idx]

        # 3. Synchronize GDS NVMe DMA: Wait for weights to land in cache slots
        expert_weight_ptrs = []
        for exp_id in active_expert_list:
            # Resolves cuFile async future; returns pre-allocated GPU VRAM tensor
            w = self.gds_cache.wait_for_expert(exp_id)
            expert_weight_ptrs.append(w)

        # Stack weights into (num_active, K, N) view for CUTLASS
        # Note: In pure C++, pass the list of device pointers directly (void** ptr_array)
        stacked_weights = torch.stack(expert_weight_ptrs, dim=0)

        # 4. Launch CUTLASS Grouped GEMM
        # Dimensions: M_i varies per expert, K = hidden_dim, N = intermediate_dim
        if grouped_gemm is not None:
            # Single-kernel variable-M execution
            grouped_out = grouped_gemm.ops.gmm(
                gathered_x, 
                stacked_weights, 
                expert_token_counts.cpu()
            )
        else:
            # Fallback block-sliced batched execution if compiled extension not present
            grouped_out = self._fallback_grouped_mm(
                gathered_x, stacked_weights, expert_token_counts
            )

        # 5. Fused Recover-LoRA Correction
        # Calculate lora_out = (x @ lora_A) @ lora_B for active experts
        lora_out = self._apply_recover_lora(
            gathered_x, active_expert_list, expert_token_counts
        )
        final_expert_out = grouped_out + lora_out

        # 6. Scatter-add back to original token ordering weighted by gating scores
        weighted_out = final_expert_out * flat_scores[token_sort_order]
        output = torch.zeros_like(flat_x)
        output.index_add_(0, orig_token_idx, weighted_out)

        return output.view(batch_size, seq_len, d_model)

    def _apply_recover_lora(
        self,
        gathered_x: torch.Tensor,
        active_experts: List[int],
        token_counts: torch.Tensor
    ) -> torch.Tensor:
        """Applies expert-specific LoRA adapters over ragged token chunks."""
        lora_out = torch.empty(
            (gathered_x.shape[0], self.intermediate_dim),
            dtype=gathered_x.dtype,
            device=self.device
        )
        offset = 0
        for i, exp_id in enumerate(active_experts):
            count = token_counts[i].item()
            if count == 0:
                continue
            chunk = gathered_x[offset : offset + count]
            # chunk @ A @ B
            res = (chunk @ self.lora_A[exp_id]) @ self.lora_B[exp_id] * self.lora_scaling
            lora_out[offset : offset + count] = res
            offset += count
        return lora_out

    def _fallback_grouped_mm(self, gathered_x, stacked_weights, token_counts):
        """Reference loop for development without CUTLASS binary."""
        out = torch.empty(
            (gathered_x.shape[0], self.intermediate_dim),
            dtype=gathered_x.dtype,
            device=self.device
        )
        offset = 0
        for i in range(stacked_weights.shape[0]):
            count = token_counts[i].item()
            if count > 0:
                out[offset : offset + count] = (
                    gathered_x[offset : offset + count] @ stacked_weights[i]
                )
                offset += count
        return out