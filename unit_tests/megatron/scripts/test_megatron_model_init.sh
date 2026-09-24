export PIVOTRL_LOGGING_PATH=${PIVOTRL_WORKSPACE}/pivotrl/unit_tests/megatron/log
export PIVOTRL_LOGGING_LEVEL=INFO
python test_megatron_model_init.py 2>&1 | tee test_megatron_model_init.log