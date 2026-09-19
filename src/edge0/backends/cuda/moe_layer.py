import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from edge0.backends.base import BaseMoELayer

try:
    import grouped_gemm
except ImportError:
    grouped_gemm = None


class CudaMoELayer(BaseMoELayer):
    """
    Standard CUDA MoE Layer.
    Uses sequential expert execution coordinated across CUDA compute/transfer streams.
    Serves as the baseline layer for the CUDA backend.
    """
    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        num_experts: int,
        top_k: int,
        backend=None,
        lora_rank: int = 16,
        device: str = "cuda:0"
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_experts=num_experts,
            top_k=top_k,
            backend=backend
        )
        self.device = torch.device(device)
        self.lora_rank = lora_rank

        # Router Gate
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False, device=self.device)

        # Recover-LoRA adapters (permanently pinned in GPU VRAM)
        self.lora_A = nn.Parameter(
            torch.randn(num_experts, hidden_dim, lora_rank, device=self.device) * 0.01
        )
        self.lora_B = nn.Parameter(
            torch.zeros(num_experts, lora_rank, intermediate_dim, device=self.device)
        )
        self.lora_scaling = 1.0 / lora_rank

    def forward(
        self,
        x: torch.Tensor,
        predicted_routing_tokens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        target_dtype = torch.bfloat16
        x = x.to(dtype=target_dtype)

        batch_size, seq_len, d_model = x.shape
        flat_x = x.view(-1, d_model)

        logits = self.gate(flat_x.to(self.gate.weight.dtype))
        routing_weights = F.softmax(logits, dim=-1)
        scores, top_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        scores = (scores / scores.sum(dim=-1, keepdim=True)).to(dtype=target_dtype)

        out = torch.zeros((flat_x.shape[0], self.intermediate_dim), dtype=target_dtype, device=self.device)

        unique_experts = torch.unique(top_indices).tolist()
        for exp_id in unique_experts:
            # Synchronize with transfer event if backend stream coordinator is present
            if self.backend and hasattr(self.backend, "transfer_ready_events"):
                if exp_id in self.backend.transfer_ready_events:
                    self.backend.compute_stream.wait_event(
                        self.backend.transfer_ready_events[exp_id]
                    )

            if self.backend and hasattr(self.backend, "expert_cache"):
                expert_weights = self.backend.expert_cache.get_expert_weights(exp_id).to(dtype=target_dtype)
            else:
                continue

            mask = (top_indices == exp_id).any(dim=-1)
            if not mask.any():
                continue

            sub_x = flat_x[mask]
            base_out = torch.matmul(sub_x, expert_weights)

            # Recover-LoRA branch
            lora_out = (
                sub_x @ self.lora_A[exp_id].to(dtype=target_dtype)
            ) @ self.lora_B[exp_id].to(dtype=target_dtype) * self.lora_scaling

            combined = base_out + lora_out
            expert_match = (top_indices[mask] == exp_id).nonzero(as_tuple=True)
            out[mask] += combined * scores[mask, expert_match[1]].unsqueeze(-1)

        return out.view(batch_size, seq_len, self.intermediate_dim)


class CUTLASSGDSMoE(BaseMoELayer):
    """
    High-Performance MoE Layer.
    Fuses GPUDirect Storage (GDS) weight prefetching with CUTLASS Grouped GEMMs
    and Recover-LoRA parameter scaling into a single execution step.
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
        super().__init__(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_experts=num_experts,
            top_k=top_k,
            backend=None
        )
        self.gds_cache = gds_cache
        self.lora_rank = lora_rank
        self.device = torch.device(device)

        # Gate router
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False, device=self.device)

        # Recover-LoRA adapters
        self.lora_A = nn.Parameter(
            torch.randn(num_experts, hidden_dim, lora_rank, device=self.device) * 0.01
        )
        self.lora_B = nn.Parameter(
            torch.zeros(num_experts, lora_rank, intermediate_dim, device=self.device)
        )
        self.lora_scaling = 1.0 / lora_rank

    def forward(
        self,
        x: torch.Tensor,
        predicted_routing_tokens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        target_dtype = torch.bfloat16
        x = x.to(dtype=target_dtype)

        batch_size, seq_len, d_model = x.shape
        flat_x = x.view(-1, d_model)
        num_tokens = flat_x.shape[0]

        # 1. Routing calculation
        logits = self.gate(flat_x.to(self.gate.weight.dtype))
        routing_weights = F.softmax(logits, dim=-1)
        scores, top_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        scores = scores / scores.sum(dim=-1, keepdim=True)

        flat_top_indices = top_indices.view(-1)
        flat_scores = scores.view(-1, 1).to(dtype=target_dtype)

        # 2. Sort tokens by expert ID
        sorted_expert_indices, token_sort_order = torch.sort(flat_top_indices)
        orig_token_idx = (
            torch.arange(num_tokens, device=self.device)
            .unsqueeze(1)
            .expand(-1, self.top_k)
            .reshape(-1)[token_sort_order]
        )

        unique_experts, expert_token_counts = torch.unique_consecutive(
            sorted_expert_indices, return_counts=True
        )
        active_expert_list = unique_experts.tolist()

        gathered_x = flat_x[orig_token_idx]

        # 3. Synchronize GDS NVMe DMA
        expert_weight_ptrs = []
        for exp_id in active_expert_list:
            w = self.gds_cache.wait_for_expert(exp_id)
            expert_weight_ptrs.append(w.to(dtype=target_dtype))

        stacked_weights = torch.stack(expert_weight_ptrs, dim=0)

        # 4. Grouped GEMM execution
        if grouped_gemm is not None:
            grouped_out = grouped_gemm.ops.gmm(
                gathered_x,
                stacked_weights,
                expert_token_counts.cpu()
            )
        else:
            grouped_out = self._fallback_grouped_mm(
                gathered_x, stacked_weights, expert_token_counts
            )

        # 5. Fused Recover-LoRA Correction
        lora_out = self._apply_recover_lora(
            gathered_x, active_expert_list, expert_token_counts, target_dtype
        )
        final_expert_out = grouped_out + lora_out

        # 6. Scatter-add back to original token ordering
        weighted_out = final_expert_out * flat_scores[token_sort_order]
        output = torch.zeros((num_tokens, self.intermediate_dim), dtype=target_dtype, device=self.device)
        output.index_add_(0, orig_token_idx, weighted_out)

        return output.view(batch_size, seq_len, self.intermediate_dim)

    def _apply_recover_lora(
        self,
        gathered_x: torch.Tensor,
        active_experts: List[int],
        token_counts: torch.Tensor,
        target_dtype: torch.dtype
    ) -> torch.Tensor:
        lora_out = torch.zeros(
            (gathered_x.shape[0], self.intermediate_dim),
            dtype=target_dtype,
            device=self.device
        )
        offset = 0
        for i, exp_id in enumerate(active_experts):
            count = token_counts[i].item()
            if count == 0:
                continue
            chunk = gathered_x[offset : offset + count]
            res = (
                chunk @ self.lora_A[exp_id].to(dtype=target_dtype)
            ) @ self.lora_B[exp_id].to(dtype=target_dtype) * self.lora_scaling
            lora_out[offset : offset + count] = res
            offset += count
        return lora_out

    def _fallback_grouped_mm(self, gathered_x, stacked_weights, token_counts):
        out = torch.zeros(
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