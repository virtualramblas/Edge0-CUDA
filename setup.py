# setup.py
import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Paths to CUDA and GPUDirect Storage header/libs
cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
cufile_include = os.path.join(cuda_home, "include")
cufile_lib = os.path.join(cuda_home, "lib64")

# Path to local CUTLASS clone/headers (downloaded or submodule)
cutlass_include = os.environ.get("CUTLASS_PATH", "/usr/local/include/cutlass")

setup(
    name="edge0",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="edge0.backends.cuda.fused_gds_gemm",
            sources=[
                "edge0/backends/cuda/c_src/cufile_expert_loader.cpp",
                "edge0/backends/cuda/c_src/cutlass_grouped_lora_kernel.cu",
            ],
            include_dirs=[cufile_include,
                        cutlass_include,
                        "edge0/backends/cuda/c_src"
            ],
            library_dirs=[cufile_lib],
            libraries=["cufile", "cuda"],  # Link GDS and CUDA drivers
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", 
                        "--use_fast_math",
                        "-gencode=arch=compute_80,code=sm_80", # Ampere
                        "-gencode=arch=compute_90,code=sm_90", # Hopper
                        ]
            }
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)