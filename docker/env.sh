#!/usr/bin/env bash
export PSRL_HOME="${PSRL_HOME:-/home/psrl}"
export PSRL_WORKSPACE="${PSRL_WORKSPACE:-/home/workspace}"
export VIRTUAL_ENV="${VIRTUAL_ENV:-/root/venv}"
export UCX_HOME="${UCX_HOME:-/root/ucx}"
export NIXL_HOME="${NIXL_HOME:-/root/nixl-install}"
export PATH="${VIRTUAL_ENV}/bin:/root/.cargo/bin:${PATH}"
export PYTHONPATH="${PSRL_HOME}${PYTHONPATH:+:${PYTHONPATH}}"
psrl_library_path="${NIXL_HOME}/lib:${NIXL_HOME}/lib64:${UCX_HOME}/lib:${UCX_HOME}/lib/ucx"
export LD_LIBRARY_PATH="${psrl_library_path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
unset psrl_library_path
export PKG_CONFIG_PATH="${UCX_HOME}/lib/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}"

if [[ -d /usr/local/cuda ]]; then
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
    export NVSHMEM_DIR="${NVSHMEM_DIR:-/root/deepep/nvshmem}"
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export HF_XET_HIGH_PERFORMANCE=1
export HF_HUB_DOWNLOAD_TIMEOUT=60
export TORCH_NCCL_TRACE_BUFFER_SIZE=10
export PYTORCH_NVML_BASED_CUDA_CHECK=1
export VERL_DATAPROTO_SERIALIZATION_METHOD=numpy
export RAY_worker_register_timeout_seconds=360
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export UCX_NET_DEVICES="bond1,bond2,bond3,bond4,bond5,bond6,bond7,bond8,mlx5_bond_1:1,mlx5_bond_4:1,mlx5_bond_3:1,mlx5_bond_2:1,mlx5_bond_7:1,mlx5_bond_6:1,mlx5_bond_8:1,mlx5_bond_5:1"
export NCCL_SOCKET_IFNAME=bond1
export NCCL_IB_HCA="mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6"
export NCCL_CUMEM_ENABLE=0
export NCCL_ALGO="allgather:^tree"
export NCCL_PRIMS_PROFILE_ENABLE=0
export NCCL_CHECK_DISABLE=1
export NCCL_LL_THRESHOLD=16384
export NCCL_IB_DISABLE=0
export NCCL_IB_SL=3
export NCCL_IB_GID_INDEX=3
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_DEBUG=VERSION
export NCCL_NET_GDR_READ=1
export NCCL_SOCKET_NTHREADS=8
export NCCL_COLLNET_ENABLE=0
export NCCL_NVLS_ENABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_PXN_DISABLE=0

export SHARP_COLL_ENABLE_SAT=0

export CUDA_DEVICE_MAX_CONNECTIONS=1

export no_proxy="${no_proxy:-${NO_PROXY:-127.0.0.1,localhost}}"
if [[ -n "${NODE_IP_LIST:-}" ]]; then
    psrl_node_ips="$(printf '%s' "${NODE_IP_LIST}" | sed -E 's/:[0-9]+//g')"
    export no_proxy="${no_proxy},${psrl_node_ips}"
    unset psrl_node_ips
fi
export NO_PROXY="${no_proxy}"
