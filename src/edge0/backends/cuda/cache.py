import os
import torch
from typing import Dict, Tuple, Optional, Union
from edge0.backends.base import BaseExpertCache

try:
    import edge0_gds_native
except ImportError:
    edge0_gds_native = None


class CudaStreamingExpertCache(BaseExpertCache):
    """
    Standard double-buffered GPU cache for streaming experts from pinned host RAM via PCIe.
    Serves as the baseline/fallback when GPUDirect Storage is not available.
    """
    def __init__(
        self,
        num_gpu_slots: int,
        expert_weight_shape: Tuple[int, ...],
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda:0"
    ):
        super().__init__()
        self.device = torch.device(device)
        self.num_slots = num_gpu_slots
        self.dtype = dtype

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

        self.slot_map: Dict[int, int] = {}
        self.events: Dict[int, torch.cuda.Event] = {}

    def prefetch_expert_async(
        self,
        expert_id: int,
        host_expert_tensor: torch.Tensor,
        transfer_stream: torch.cuda.Stream
    ) -> torch.cuda.Event:
        """Transfers an expert asynchronously from pinned host memory to GPU."""
        slot_idx = expert_id % self.num_slots
        self.slot_map[expert_id] = slot_idx

        self.host_staging[slot_idx].copy_(host_expert_tensor, non_blocking=True)

        event = torch.cuda.Event()
        with torch.cuda.stream(transfer_stream):
            self.device_slots[slot_idx].copy_(
                self.host_staging[slot_idx],
                non_blocking=True
            )
            event.record(transfer_stream)

        self.events[expert_id] = event
        return event

    def wait_for_expert(self, expert_id: int) -> torch.Tensor:
        slot_idx = self.slot_map.get(expert_id, expert_id % self.num_slots)
        return self.device_slots[slot_idx]

    def get_expert_weights(self, expert_id: int) -> torch.Tensor:
        return self.wait_for_expert(expert_id)


class GDSStreamingExpertCache(BaseExpertCache):
    """
    GPUDirect Storage (cuFile) Cache for direct NVMe -> GPU VRAM streaming.
    Bypasses CPU host RAM and coordinates directly with the lookahead prerouter.
    """
    def __init__(
        self,
        num_slots: int,
        expert_shape: Tuple[int, ...],
        loader: Optional[Union[object, str]] = None,
        file_path: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda:0"
    ):
        super().__init__()
        self.num_slots = num_slots
        self.device = torch.device(device)
        self.dtype = dtype

        # Resolve loader: allows passing either an instance or a file path string
        resolved_path = file_path if file_path is not None else (loader if isinstance(loader, str) else None)
        if resolved_path is not None:
            if edge0_gds_native is None:
                raise RuntimeError("edge0_gds_native extension is not compiled.")
            self.loader = edge0_gds_native.GDSNativeLoader(resolved_path)
        elif loader is not None:
            self.loader = loader
        else:
            raise ValueError("Either 'loader' or 'file_path' must be provided to GDSStreamingExpertCache.")

        # Pre-allocate pinned GPU slots directly in VRAM
        self.gpu_slots = torch.zeros(
            (num_slots, *expert_shape),
            dtype=dtype,
            device=self.device
        )

        self.io_futures: Dict[int, object] = {}
        self.slot_map: Dict[int, int] = {}
        self.expert_meta: Dict[int, Tuple[int, int]] = {}

    def register_expert_offset(self, expert_id: int, offset: int, num_bytes: int):
        """Maps an expert ID to its binary file offset and byte length."""
        self.expert_meta[expert_id] = (offset, num_bytes)

    def trigger_prerouter_load(self, expert_id: int):
        """Called 1-2 layers ahead by the lookahead prerouter."""
        slot_idx = expert_id % self.num_slots
        self.slot_map[expert_id] = slot_idx

        if hasattr(self.loader, "load_expert") and expert_id in self.expert_meta:
            offset, num_bytes = self.expert_meta[expert_id]
            self.loader.load_expert(self.gpu_slots[slot_idx], offset, num_bytes)
        elif hasattr(self.loader, "read_expert_to_gpu_async"):
            future = self.loader.read_expert_to_gpu_async(
                expert_id=expert_id,
                target_gpu_tensor=self.gpu_slots[slot_idx]
            )
            self.io_futures[expert_id] = future

    def wait_for_expert(self, expert_id: int) -> torch.Tensor:
        """Called when the MoE compute kernel is about to execute."""
        if expert_id in self.io_futures:
            self.io_futures[expert_id].get()
            del self.io_futures[expert_id]

        slot_idx = self.slot_map.get(expert_id, expert_id % self.num_slots)
        return self.gpu_slots[slot_idx]