# First run the target script
# Then run the initiator script
# export UCX_LOG_LEVEL=debug

TARGET_IP=28.49.196.77
PIVOTRL_WORKSPACE=${PIVOTRL_WORKSPACE}/pivotrl_agent
CONDA_ENV_FILE=${PIVOTRL_WORKSPACE}/../activate
CONDA_ENV_NAME=pivotrl-lhy-agent
GPU_ID=-1
IP=${TARGET_IP}

source ${CONDA_ENV_FILE} 
conda activate ${CONDA_ENV_NAME} 
# export UCX_LOG_LEVEL=debug
# export NIXL_LOG_LEVEL=debug
export UCX_NET_DEVICES="bond1,bond2,bond3,bond4,bond5,bond6,bond7,bond8,mlx5_bond_1:1,mlx5_bond_4:1,mlx5_bond_3:1,mlx5_bond_2:1,mlx5_bond_7:1,mlx5_bond_6:1,mlx5_bond_8:1,mlx5_bond_5:1" 
echo "UCX_NET_DEVICES: ${UCX_NET_DEVICES}"

PYTHONUNBUFFERED=1 python ${PIVOTRL_WORKSPACE}/pivotrl/unit_tests/nixl/test_send_recv.py --ip ${IP} --mode initiator --cuda ${GPU_ID}
# PYTHONUNBUFFERED=1 python ${PIVOTRL_WORKSPACE}/pivotrl/unit_tests/nixl/test_send_recv_model.py --ip ${IP} --mode initiator --cuda ${GPU_ID} --model_path ${PIVOTRL_WORKSPACE}/models/Qwen2.5-Math-7B
