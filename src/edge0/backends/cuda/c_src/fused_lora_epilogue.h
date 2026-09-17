#pragma once
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/array.h>
#include <cutlass/functional.h>
#include <cutlass/epilogue/thread/linear_combination.h>

namespace edge0 {
namespace cuda {

/**
 * Custom Epilogue Functor that computes:
 * D = alpha * (A @ B_base) + lora_scaling * (X_lora @ B_lora) + beta * C
 * 
 * Computes the low-rank B_lora GEMV/GEMM directly on register accumulators
 * before committing to global memory.
 */
template <
    typename ElementOutput_,
    typename ElementAccumulator_,
    typename ElementCompute_,
    int ElementsPerAccess,
    int LoRARank = 16
>
class FusedRecoverLoRAEpilogue {
public:
    using ElementOutput = ElementOutput_;
    using ElementAccumulator = ElementAccumulator_;
    using ElementCompute = ElementCompute_;
    static int const kElementsPerAccess = ElementsPerAccess;
    static int const kRank = LoRARank;

    using FragmentOutput = cutlass::Array<ElementOutput, kElementsPerAccess>;
    using FragmentAccumulator = cutlass::Array<ElementAccumulator, kElementsPerAccess>;
    using ComputeFragment = cutlass::Array<ElementCompute, kElementsPerAccess>;

    struct Params {
        ElementCompute alpha;
        ElementCompute beta;
        ElementCompute lora_scaling;
        ElementCompute const* lora_intermediate_ptr; // Points to (X @ A_lora), shape [M, r]
        ElementCompute const* lora_B_weight_ptr;      // Points to B_lora, shape [r, N]
        int64_t lora_B_stride_k;                     // Stride between rank slices

        CUTLASS_HOST_DEVICE
        Params(
            ElementCompute alpha_ = ElementCompute(1),
            ElementCompute beta_ = ElementCompute(0),
            ElementCompute lora_scale = ElementCompute(1.0f / LoRARank),
            ElementCompute const* lora_inter = nullptr,
            ElementCompute const* lora_B = nullptr,
            int64_t stride_k = 0
        ) : alpha(alpha_), beta(beta_), lora_scaling(lora_scale),
            lora_intermediate_ptr(lora_inter), lora_B_weight_ptr(lora_B),
            lora_B_stride_k(stride_k) {}
    };

private:
    Params params_;

public:
    CUTLASS_HOST_DEVICE
    FusedRecoverLoRAEpilogue(Params const& params) : params_(params) {}

    CUTLASS_HOST_DEVICE
    bool is_source_needed() const {
        return params_.beta != ElementCompute(0);
    }

    /// Primary operator applying the fused fusion
    CUTLASS_HOST_DEVICE
    void operator()(
        FragmentOutput &accum_out,
        FragmentAccumulator const &accum_base,
        FragmentOutput const &source,
        int row_idx,
        int col_idx
    ) const {
        ComputeFragment intermediate;

        // 1. Scale base accumulator: alpha * BaseGEMM
        #pragma unroll
        for (int i = 0; i < kElementsPerAccess; ++i) {
            intermediate[i] = params_.alpha * ElementCompute(accum_base[i]);
        }

        // 2. Fused LoRA Dot-Product directly in registers:
        // lora_out = sum_{k=0}^{Rank-1} (X @ A)[row, k] * B[k, col]
        if (params_.lora_intermediate_ptr && params_.lora_B_weight_ptr) {
            #pragma unroll
            for (int i = 0; i < kElementsPerAccess; ++i) {
                int global_col = col_idx + i;
                ElementCompute lora_acc = ElementCompute(0);

                #pragma unroll
                for (int r = 0; r < kRank; ++r) {
                    ElementCompute act_a = params_.lora_intermediate_ptr[row_idx * kRank + r];
                    ElementCompute weight_b = params_.lora_B_weight_ptr[r * params_.lora_B_stride_k + global_col];
                    lora_acc += act_a * weight_b;
                }

                // Add scaled Recover-LoRA to the base GEMM output
                intermediate[i] += params_.lora_scaling * lora_acc;
            }
        }

        // 3. Optional residual addition (beta * C) and cast to output dtype
        #pragma unroll
        for (int i = 0; i < kElementsPerAccess; ++i) {
            if (params_.beta != ElementCompute(0)) {
                intermediate[i] += params_.beta * ElementCompute(source[i]);
            }
            accum_out[i] = ElementOutput(intermediate[i]);
        }
    }
};

} // namespace cuda
} // namespace edge0