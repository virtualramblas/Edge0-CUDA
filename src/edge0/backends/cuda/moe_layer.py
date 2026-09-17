import torch
import torch.nn as nn
import torch.nn.functional as F

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