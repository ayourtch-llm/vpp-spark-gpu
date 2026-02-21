# SPDX-License-Identifier: Apache-2.0
# gpu_classify plugin — GPU-accelerated packet classifier for VPP.
#
# Requires the NVIDIA CUDA toolkit.
# Target: NVIDIA Blackwell GB10 (DGX Spark, NVLink-C2C), sm_100.
#
# CUDA toolkit installation hint (Ubuntu 24.04 aarch64/SBSA):
#   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-keyring_1.1-1_all.deb
#   dpkg -i cuda-keyring_1.1-1_all.deb && apt-get update
#   apt-get install -y cuda-toolkit-12-8
# Then build VPP with:
#   export PATH=/usr/local/cuda-12.8/bin:$PATH
#   make build

# ------------------------------------------------------------------ #
# Locate nvcc                                                          #
# ------------------------------------------------------------------ #

if (NOT DEFINED CMAKE_CUDA_COMPILER)
  find_program (CMAKE_CUDA_COMPILER NAMES nvcc
    PATHS
      /usr/local/cuda/bin
      /usr/local/cuda-12.8/bin
      /usr/local/cuda-13.0/bin
    NO_DEFAULT_PATH)
endif ()

if (NOT CMAKE_CUDA_COMPILER)
  message (WARNING
    "nvcc not found — gpu_classify plugin disabled. "
    "Install cuda-toolkit-12-8 and add its bin/ to PATH.")
  return ()
endif ()

message (STATUS "gpu_classify: using nvcc at ${CMAKE_CUDA_COMPILER}")
enable_language (CUDA)
find_package (CUDAToolkit REQUIRED)

# ------------------------------------------------------------------ #
# CUDA kernel static library                                           #
#                                                                      #
# Compiled by nvcc as a self-contained (non-separable) translation    #
# unit. POSITION_INDEPENDENT_CODE ensures -Xcompiler -fPIC is passed  #
# so the resulting .a can be linked into the plugin .so.              #
# ------------------------------------------------------------------ #

add_library (gpu_classify_cuda_kernel STATIC
  gpu_classify_kernel.cu
)

set_target_properties (gpu_classify_cuda_kernel PROPERTIES
  CUDA_ARCHITECTURES        "100"    # Blackwell GB10 = sm_100
  CUDA_SEPARABLE_COMPILATION OFF     # single-TU build, no dlink step
  CUDA_STANDARD             17
  POSITION_INDEPENDENT_CODE ON
)

target_include_directories (gpu_classify_cuda_kernel PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_BINARY_DIR}
  ${CUDAToolkit_INCLUDE_DIRS}
)

# ------------------------------------------------------------------ #
# VPP plugin (standard C sources)                                     #
# ------------------------------------------------------------------ #

add_vpp_plugin (gpu_classify
  SOURCES
    gpu_classify.c
    gpu_classify_node.c

  LINK_LIBRARIES
    gpu_classify_cuda_kernel
    CUDA::cudart              # dynamic cudart resolved at plugin load

  COMPONENT vpp-plugin-core
)

# Expose the plugin's own header directory to the plugin target so
# #include <gpu_classify/gpu_classify.h> resolves correctly.
target_include_directories (gpu_classify_plugin PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CUDAToolkit_INCLUDE_DIRS}
)
