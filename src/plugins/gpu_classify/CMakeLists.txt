# SPDX-License-Identifier: Apache-2.0
# gpu_classify plugin — GPU-accelerated packet classifier for VPP.
#
# Requires the NVIDIA CUDA toolkit.
# Target: NVIDIA Blackwell GB10 (DGX Spark, NVLink-C2C), sm_100.
#
# CUDA toolkit installation hint (Ubuntu 24.04 aarch64/SBSA):
#   curl -fsSL -o /tmp/cuda-keyring.deb \
#     https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-keyring_1.1-1_all.deb
#   dpkg -i /tmp/cuda-keyring.deb && apt-get update
#   apt-get install -y cuda-toolkit-12-8
# Then simply run: make build-release   (no PATH changes needed)

# ------------------------------------------------------------------ #
# Locate nvcc — fully automatic, no PATH setup required              #
#                                                                      #
# file(GLOB ... CONFIGURE_DEPENDS) is re-evaluated on every           #
# "cmake --build" invocation.  When nvcc appears in a standard        #
# location (e.g. after installing cuda-toolkit-12-8), cmake detects   #
# the change and retriggers configuration automatically — the user     #
# just needs to re-run "make build-release".                          #
# ------------------------------------------------------------------ #

file(GLOB _nvcc_candidates LIST_DIRECTORIES false CONFIGURE_DEPENDS
  "/usr/local/cuda/bin/nvcc"
  "/usr/local/cuda-*/bin/nvcc"
)

if (_nvcc_candidates)
  # If multiple toolkits are installed, prefer the newest version.
  list(SORT _nvcc_candidates ORDER DESCENDING)
  list(GET _nvcc_candidates 0 _nvcc_path)
endif ()

# Fallback: search directories listed in PATH (handles non-standard installs).
if (NOT _nvcc_path)
  find_program(_nvcc_path_tmp NAMES nvcc)
  if (_nvcc_path_tmp)
    set(_nvcc_path "${_nvcc_path_tmp}")
  endif ()
  unset(_nvcc_path_tmp CACHE)   # don't let find_program cache a stale result
endif ()

if (NOT _nvcc_path)
  message (WARNING
    "nvcc not found — gpu_classify plugin disabled. "
    "Install cuda-toolkit-12-8 and re-run make build-release "
    "(cmake will detect the toolkit automatically).")
  return ()
endif ()

# Force-update CMAKE_CUDA_COMPILER so a stale NOTFOUND cache entry
# from a previous configure (run before the toolkit was installed)
# does not prevent the plugin from being built.
set (CMAKE_CUDA_COMPILER "${_nvcc_path}" CACHE FILEPATH
  "CUDA compiler (nvcc)" FORCE)

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
