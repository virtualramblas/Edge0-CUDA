import torch
from edge0.backends.base import BaseExpertCache

class CudaStreamingExpertCache(BaseExpertCache):
    """
    Double-buffered GPU cache for streaming experts from host/SSD.
    """
    def __init__(
        self,
        num_gpu_slots: int,
        expert_weight_shape: Tuple[int, ...],
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda:0"
    ):
        self.device = torch.device(device)
        self.num_slots = num_gpu_slots
        
        # Pre-allocate pinned memory staging buffer on Host for zero-copy PCIe DMA
        self.host_staging = torch.empty(
            (num_gpu_slots, *expert_weight_shape),
            dtype=dtype,
            pin_memory=True
        )
        
        # Pre-allocate expert weight slots in GPU VRAM
        self.device_slots = torch.empty(
            (num_gpu_slots, *expert_weight_shape),
            dtype=dtype,
            device=self.device
        )
        
        # Mapping from expert_id -> slot_idx
        self.slot_map: Dict[int, int] = {}
        self.events: Dict[int, torch.cuda.Event] = {}

    def prefetch_expert_async(
        self,
        expert_id: int,
        host_expert_tensor: torch.Tensor,
        transfer_stream: torch.cuda.Stream
    ) -> torch.cuda.Event:
        """
        Transfers an expert asynchronously from pinned host memory to GPU.
        """
        slot_idx = expert_id % self.num_slots
        self.slot_map[expert_id] = slot_idx

        # Copy data into pinned staging buffer if not already pinned
        self.host_staging[slot_idx].copy_(host_expert_tensor, non_blocking=True)

        event = torch.cuda.Event()
        with torch.cuda.stream(transfer_stream):
            # Asynchronous DMA Host -> Device copy over PCIe
            self.device_slots[slot_idx].copy_(
                self.host_staging[slot_idx],
                non_blocking=True
            )
            event.record(transfer_stream)

        self.events[expert_id] = event
        return event

    def get_expert_weights(self, expert_id: int) -> torch.Tensor:
        slot_idx = self.slot_map[expert_id]
        return self.device_slots[slot_idx]