#include <torch/extension.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include "fused_lora_epilogue.h"

// Define data types
using ElementInput = cutlass::bfloat16_t;
using ElementOutput = cutlass::bfloat16_t;
using ElementAccumulator = float;
using ElementCompute = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor; // Or RowMajor depending on GDS storage
using LayoutC = cutlass::layout::RowMajor;

static const int kLoRARank = 16;
static const int kElementsPerAccess = 8; // 128-bit vectorization for BF16

// Custom Epilogue specialization
using CustomEpilogue = edge0::cuda::FusedRecoverLoRAEpilogue<
    ElementOutput,
    ElementAccumulator,
    ElementCompute,
    kElementsPerAccess,
    kLoRARank
>;

// Threadblock & Warp Tile configurations (Ampere SM80 / Hopper SM90)
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

// Construct Grouped GEMM Kernel using CUTLASS Device API
using GemmGroupedKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    ElementInput, LayoutA, cutlass::ComplexTransform::kNone, 8,
    ElementInput, LayoutB, cutlass::ComplexTransform::kNone, 8,
    ElementOutput, LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80, // Targets NVIDIA Ampere / Hopper / Blackwell
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    CustomEpilogue,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    3 // 3-stage shared memory pipeline
>::GemmKernel;

using EpilogueGrouped = cutlass::gemm::device::GemmGrouped<GemmGroupedKernel>;

// Host launch wrapper
void launch_fused_grouped_gemm_lora(
    torch::Tensor x,                       // [Total_Tokens, Hidden_Dim]
    std::vector<torch::Tensor> base_w_ptrs,// List of GDS cached GPU weights
    torch::Tensor lora_A,                  // [E, Hidden_Dim, Rank]
    torch::Tensor lora_B,                  // [E, Rank, Intermediate_Dim]
    torch::Tensor token_counts,            // [E]
    torch::Tensor out                      // [Total_Tokens, Intermediate_Dim]
) {
    int problem_count = base_w_ptrs.size();
    
    // Set up problem arguments (ptr-array mode)
    std::vector<cutlass::gemm::GemmCoord> problem_sizes;
    int offset = 0;
    for (int i = 0; i < problem_count; ++i) {
        int m_i = token_counts[i].item<int>();
        int k = x.size(1);
        int n = out.size(1);
        problem_sizes.push_back({m_i, n, k});
    }

    // Allocate & initialize CUTLASS grouped args on device
    // (Pointers, strides, and LoRA custom epilogue params)
    typename EpilogueGrouped::Arguments args(
        problem_sizes.data(),
        problem_count,
        /* threadblock count */ 24,
        /* Epilogue params */
        typename CustomEpilogue::Params(
            1.0f, // alpha
            0.0f, // beta
            1.0f / kLoRARank, // lora_scaling
            reinterpret_cast<ElementCompute const*>(x.data_ptr()),
            reinterpret_cast<ElementCompute const*>(lora_B.data_ptr()),
            out.size(1)
        ),
        /* Pointer arrays */ ...
    );

    EpilogueGrouped gemm_op;
    cutlass::Status status = gemm_op.initialize(args);
    TORCH_CHECK(status == cutlass::Status::kSuccess, "Failed to initialize CUTLASS Grouped GEMM");

    status = gemm_op();
    TORCH_CHECK(status == cutlass::Status::kSuccess, "Failed to run CUTLASS Grouped GEMM");
}