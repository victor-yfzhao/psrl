import logging
from contextlib import contextmanager

import torch

psrl_logger = logging.getLogger(__file__)

def apply_megatron_distrib_optimizer_patch():
    """
    Patch Megatron's DistributedOptimizer._set_main_param_and_optimizer_states to skip
    non-tensor entries (e.g. 'padding': False) in the loaded optimizer bucket state.

    Bug context: When using dp_reshardable checkpoint format together with                            
    use_precision_aware_optimizer=True, the loaded optimizer state dict contains a                    
    'padding' key with a bool value (from LocalNonpersistentObject). The precision-aware              
    code path naively iterates over ALL keys and calls set_scaled_state on the bool,                  
    causing: AttributeError: 'bool' object has no attribute 'dtype'.                                  

    This patch filters out non-Tensor values before processing.                                       
    """
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer                        

    if getattr(DistributedOptimizer, "_psrl_set_main_param_patched", False):                          
        return

    original_fn = DistributedOptimizer._set_main_param_and_optimizer_states                           

    def _patched_set_main_param_and_optimizer_states(self, model_param, tensors):                     
        # NOTE(claude): Filter out non-tensor metadata keys (e.g. 'padding': False)
        # that are preserved from DCP LocalNonpersistentObject during checkpoint loading
        filtered_tensors = {k: v for k, v in tensors.items() if isinstance(v, torch.Tensor)}
        return original_fn(self, model_param, filtered_tensors)

    DistributedOptimizer._set_main_param_and_optimizer_states = (
        _patched_set_main_param_and_optimizer_states
    )
    DistributedOptimizer._psrl_set_main_param_patched = True
    psrl_logger.info(
        "[apply_megatron_distrib_optimizer_patch] Patched "
        "DistributedOptimizer._set_main_param_and_optimizer_states to skip non-tensor entries."
    )

def apply_tms_patch():
    from torch_memory_saver.entrypoint import _TorchMemorySaverImpl

    _TAG_DEFAULT = "default"

    @contextmanager
    def _with_region_config_patch(self, tag: str, enable_cpu_backup: bool):
        # assert not self._binary_wrapper.cdll.tms_get_interesting_region()
        original_enable_cpu_backup = self._binary_wrapper.cdll.tms_get_enable_cpu_backup()
        original_interesting_region = self._binary_wrapper.cdll.tms_get_interesting_region()

        self._binary_wrapper.set_config(tag=tag, interesting_region=True, enable_cpu_backup=enable_cpu_backup)
        try:
            yield
        finally:
            assert self._binary_wrapper.cdll.tms_get_interesting_region()
            self._binary_wrapper.set_config(
                tag=_TAG_DEFAULT,
                interesting_region=original_interesting_region,
                enable_cpu_backup=original_enable_cpu_backup,
            )

    _TorchMemorySaverImpl._with_region_config = _with_region_config_patch
