import os
import torch
from typing import Dict, Tuple, Optional
import kvikio
from kvikio.cufile import IOFuture  # 📍 CORRECTED FUTURE IMPORT

class GDSExpertLoader:
    """
    GPUDirect Storage (cuFile) loader that DMA-transfers expert weights
    directly from NVMe filesystems into GPU VRAM buffers.
    """
    def __init__(
        self,
        checkpoint_path: str,
        expert_manifest: Dict[int, Tuple[int, int]], # expert_id -> (file_offset, num_bytes)
        device: torch.device = torch.device("cuda:0")
    ):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.expert_manifest = expert_manifest

        # Verify if GPUDirect Storage driver (nvidia-fs) is active
        self.gds_active = kvikio.defaults.compat_mode() is False
        print(f"[*] Edge0 GDS Loader initialized. Driver Direct DMA Active: {self.gds_active}")

        # Open file with direct IO semantics (O_DIRECT)
        self.file_handle = kvikio.CuFile(self.checkpoint_path, flags="r")

    def read_expert_to_gpu_async(
        self,
        expert_id: int,
        target_gpu_tensor: torch.Tensor
    ) -> IOFuture:  # 📍 CORRECTED TYPE ANNOTATION
        """
        Asynchronously streams expert bytes directly into GPU tensor memory via cuFile.
        Bypasses CPU host RAM completely.
        """
        assert target_gpu_tensor.is_cuda, "Target tensor must reside in GPU memory."
        offset, num_bytes = self.expert_manifest[expert_id]

        # Ensure the receiving buffer view matches the slice size
        target_slice = target_gpu_tensor.view(torch.uint8)[:num_bytes]

        # Non-blocking cuFile DMA transfer directly from NVMe into GPU VRAM
        future = self.file_handle.pread(
            buf=target_slice,
            size=num_bytes,
            file_offset=offset
        )
        return future

    def close(self):
        if hasattr(self, "file_handle"):
            self.file_handle.close()