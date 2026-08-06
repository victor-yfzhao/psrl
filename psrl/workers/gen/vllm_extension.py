import gc
import logging
import os
import socket
import time
from contextlib import nullcontext
from copy import copy, deepcopy

import ray
import torch
from omegaconf import DictConfig

try:
    # for torch 2.5+
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor

# from vllm.v1.worker.gpu_worker import Worker
from verl.utils.device import get_device_id

# from vllm.config import get_current_vllm_config
# from vllm.platforms import current_platform
from verl.utils.fs import copy_to_local
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader
from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm.model_executor.model_loader import get_model_loader
from vllm.v1.core.kv_cache_utils import estimate_max_model_len

from psrl.utils.common.nixl_names import NIXL_META_SERVER_NAME
from psrl.utils.common.worker_naming import gen_client_name, ps_agent_name
from psrl.utils.converter import create_parameter_mapping
from psrl.utils.converter.vllm_converter import convert_vllm_inplace
from psrl.utils.nixl import (
    NIXLClientType,
    NIXLInterface,
    NIXLStorageClient,
)

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


"""
class NIXLWorker(Worker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # We do the device initialization here before building the NIXL client 
        self.device = torch.device(f"cuda:{self.local_rank}")
        current_platform.set_device(self.device)
        current_platform.check_if_supports_dtype(self.model_config.dtype)
        
        vllm_config = get_current_vllm_config()
        assert vllm_config.additional_config is not None, "additional_config must be provided when using NIXL"
        assert vllm_config.additional_config.get("nixl_config") is not None, \
            "nixl_config must be provided when using NIXL"
        assert vllm_config.additional_config.get("nixl_interface") is not None, \
            "nixl_interface must be provided when using NIXL"
        assert vllm_config.additional_config.get("instance_id") is not None, \
            "instance_id must be provided when using NIXL"
        self.nixl_config = vllm_config.additional_config.get("nixl_config")
        self.nixl_interface = vllm_config.additional_config.get("nixl_interface")
        self.instance_id = vllm_config.additional_config.get("instance_id")
        
        assert hasattr(self, "init_nixl_client"), "init_nixl_client must be provided when using NIXL"
        self.init_nixl_client(self.nixl_config, self.nixl_interface, self.instance_id)
"""


class vLLMWorkerExtension:
    @staticmethod
    def _maybe_tms_weights_region():
        """Tag temporary GPU allocations as weights when the TMS vLLM patch is active."""
        if not os.environ.get("PSRL_VLLM_PATCHES", "").startswith("TMS"):
            return nullcontext()
        try:
            from torch_memory_saver import torch_memory_saver
        except ImportError:
            return nullcontext()
        return torch_memory_saver.region(tag="weights")

    def preload_weights_to_cpu_cache(self, weights_path: str | None = None, load_format: str | None = None) -> int:
        """
        Preload checkpoint weights into CPU memory for reward-model wake up.

        The cache stores checkpoint-format weights, so wake up can remap GPU
        parameter memory first and then call `model.load_weights` without
        reading checkpoint files again.
        """
        try:
            model = self.model_runner.model
            if isinstance(model, CUDAGraphWrapper):
                model = model.unwrap()

            model_config = copy(self.model_config)
            if weights_path is not None:
                model_config.model = copy_to_local(weights_path)

            load_config = deepcopy(self.model_runner.load_config)
            if load_format is not None:
                # Reward models are initialized with dummy weights on GPU, but
                # the CPU cache must come from the real checkpoint.
                load_config.load_format = "auto" if str(load_format).startswith("dummy") else load_format

            model_loader = get_model_loader(load_config)
            if not hasattr(model_loader, "get_all_weights"):
                raise NotImplementedError(
                    f"CPU weight cache does not support load format `{load_config.load_format}`."
                )

            cache: list[tuple[str, torch.Tensor]] = []
            for name, tensor in model_loader.get_all_weights(model_config, model):
                if isinstance(tensor, DTensor):
                    tensor = tensor.full_tensor()
                cache.append((name, tensor.detach().cpu().clone()))

            self._psrl_cpu_weight_cache = cache
            self._psrl_reward_weight_cache_state = {
                "ready": True,
                "source": "checkpoint",
                "checkpoint_tensor_count": len(cache),
            }
            psrl_logger.info("Preloaded %d reward model tensors to CPU cache.", len(cache))
            return len(cache)
        except Exception as e:
            raise ValueError(f"Error in vLLMWorkerExtension.preload_weights_to_cpu_cache: {e}") from e

    def load_weights_from_cpu_cache(self, blocking: bool = True):
        """
        Load reward-model weights from the worker-local CPU cache to GPU.
        """
        total_start = time.perf_counter()
        gpu_copy_elapsed_s = 0.0
        empty_cache_elapsed_s = 0.0
        arena_snapshot_elapsed_s = 0.0
        cache_release_elapsed_s = 0.0
        arena_restore_bytes = 0
        cache_source = "checkpoint"
        cache_transitioned = False
        model_validation_elapsed_s = 0.0
        arena_handle = getattr(self.model_runner, "_psrl_weight_arena_handle", None)
        arena_cache = getattr(self, "_psrl_weight_arena_cpu_cache", None)
        try:
            model = self.model_runner.get_model()
            cache_state = getattr(self, "_psrl_reward_weight_cache_state", None)
            if arena_handle is not None:
                validation_start = time.perf_counter()
                arena_handle.assert_module_weights_in_arena(model)
                model_validation_elapsed_s += time.perf_counter() - validation_start

            if arena_cache is not None:
                if arena_handle is None:
                    raise RuntimeError("Reward arena CPU cache exists without a weight arena handle.")
                if cache_state is None or not cache_state.get("ready") or cache_state.get("source") != "arena":
                    raise RuntimeError(f"Reward arena CPU cache has invalid state: {cache_state!r}.")
                from psrl.utils.weight_arena import restore_weight_arena_from_cpu

                cache_source = "arena"
                torch.cuda.synchronize()
                gpu_copy_start = time.perf_counter()
                arena_restore_bytes = restore_weight_arena_from_cpu(arena_handle, arena_cache)
                gpu_copy_elapsed_s = time.perf_counter() - gpu_copy_start
                validation_start = time.perf_counter()
                arena_handle.assert_module_weights_in_arena(model)
                model_validation_elapsed_s += time.perf_counter() - validation_start
                loaded_params = {"__psrl_weight_arena_restore__"}
            else:
                if not hasattr(self, "_psrl_cpu_weight_cache"):
                    raise RuntimeError("CPU weight cache is not initialized.")
                if cache_state is None or not cache_state.get("ready") or cache_state.get("source") != "checkpoint":
                    raise RuntimeError(f"Reward checkpoint CPU cache has invalid state: {cache_state!r}.")
                if arena_handle is not None and not blocking:
                    raise RuntimeError("Reward arena snapshot requires blocking=True.")

                current_device = torch.cuda.current_device()

                def cached_weights_generator():
                    for name, tensor in self._psrl_cpu_weight_cache:
                        yield (name, tensor.to(current_device, non_blocking=True))

                torch.cuda.synchronize()
                gpu_copy_start = time.perf_counter()
                with self._maybe_tms_weights_region():
                    loaded_params = model.load_weights(weights=cached_weights_generator())
                if loaded_params is None:
                    raise RuntimeError("Reward checkpoint loader returned no load result.")
                if blocking:
                    torch.cuda.synchronize()
                    gpu_copy_elapsed_s = time.perf_counter() - gpu_copy_start
                    if arena_handle is not None:
                        validation_start = time.perf_counter()
                        arena_handle.assert_module_weights_in_arena(model)
                        model_validation_elapsed_s += time.perf_counter() - validation_start
                    # Release checkpoint-layout CPU-to-GPU staging tensors before
                    # taking the persistent arena snapshot.
                    empty_cache_start = time.perf_counter()
                    aggressive_empty_cache(force_sync=True)
                    empty_cache_elapsed_s = time.perf_counter() - empty_cache_start

                if arena_handle is not None:
                    from psrl.utils.weight_arena import snapshot_weight_arena_to_cpu

                    arena_config = (self.model_runner.vllm_config.additional_config or {}).get(
                        "psrl_nixl_weight_arena", {}
                    )
                    pin_memory = bool(arena_config.get("reward_cpu_cache_pin_memory", False))
                    snapshot_start = time.perf_counter()
                    self._psrl_weight_arena_cpu_cache = snapshot_weight_arena_to_cpu(
                        arena_handle,
                        pin_memory=pin_memory,
                    )
                    arena_snapshot_elapsed_s = time.perf_counter() - snapshot_start
                    arena_bytes = sum(
                        tensor.numel() * tensor.element_size()
                        for tensor in self._psrl_weight_arena_cpu_cache
                    )
                    self._psrl_reward_weight_cache_state = {
                        "ready": True,
                        "source": "arena",
                        "arena_count": len(self._psrl_weight_arena_cpu_cache),
                        "arena_bytes": arena_bytes,
                        "pin_memory": pin_memory,
                    }
                    cache_transitioned = True
                    release_start = time.perf_counter()
                    del self._psrl_cpu_weight_cache
                    gc.collect()
                    cache_release_elapsed_s = time.perf_counter() - release_start
        except Exception as e:
            raise ValueError(f"Error in vLLMWorkerExtension.load_weights_from_cpu_cache: {e}") from e
        elapsed_s = time.perf_counter() - total_start
        arena_count = len(arena_handle.arenas) if arena_handle is not None else 0
        arena_bytes = sum(arena.numel() * arena.element_size() for arena in arena_handle.arenas) if arena_handle else 0
        self._psrl_worker_stage_timings = getattr(self, "_psrl_worker_stage_timings", {})
        self._psrl_worker_stage_timings["load_weights_from_cpu_cache"] = {
            "cache_source": cache_source,
            "gpu_copy_elapsed_s": gpu_copy_elapsed_s,
            "empty_cache_elapsed_s": empty_cache_elapsed_s,
            "arena_snapshot_elapsed_s": arena_snapshot_elapsed_s,
            "cache_release_elapsed_s": cache_release_elapsed_s,
            "arena_restore_bytes": arena_restore_bytes,
            "arena_count": arena_count,
            "arena_bytes": arena_bytes,
            "cache_transitioned": cache_transitioned,
            "model_validation_elapsed_s": model_validation_elapsed_s,
            "elapsed_s": elapsed_s,
        }
        psrl_logger.warning(
            "[VLLM_SLEEP_WAKE_TIMING] scope=tp_worker operation=wake "
            "stage=load_weights_from_cpu_cache host=%s rank=%s tp_rank=%s pid=%s "
            "cache_source=%s gpu_copy_elapsed_s=%.6f empty_cache_elapsed_s=%.6f "
            "arena_snapshot_elapsed_s=%.6f cache_release_elapsed_s=%.6f "
            "arena_restore_bytes=%d arena_count=%d arena_bytes=%d cache_transitioned=%s "
            "model_validation_elapsed_s=%.6f elapsed_s=%.6f",
            socket.gethostname(),
            self.get_instance_local_rank(),
            self.get_instance_local_tp_rank(),
            os.getpid(),
            cache_source,
            gpu_copy_elapsed_s,
            empty_cache_elapsed_s,
            arena_snapshot_elapsed_s,
            cache_release_elapsed_s,
            arena_restore_bytes,
            arena_count,
            arena_bytes,
            cache_transitioned,
            model_validation_elapsed_s,
            elapsed_s,
        )
        return loaded_params

    def load_weights(self, weights, blocking: bool = True):
        """
        Load weights into the vLLM model runner.

        This method rebuilds the weights using the provided function and arguments stemming from `reduce_tensor` calls,
        transfers them to the current CUDA device, and loads them into the vLLM model runner.
        If the weight is a DTensor, it converts it to a full tensor before loading.
        If `blocking` is True, it ensures that all operations are completed before returning.
        If an error occurs during the process, it logs the error and returns None.

        Args:
            weights (List[tuple]): A list of tuples where each tuple contains:
                - name (str): The name of the weight.
                - handle (tuple): A tuple containing the function and its arguments to rebuild the weight.
            blocking (bool): If True, will block until all operations are completed.

        Returns:
            loaded_params: The loaded parameters from the model runner.

        Raises:
            Exception: If there is an error during the loading process.
        """
        try:

            def rebuild_weights_generator():
                current_device = torch.cuda.current_device()
                for name, handle in weights:
                    func, args = handle
                    list_args = list(args)
                    # CPU bundle: (type(tensor), storage, metadata)
                    if len(list_args) == 3:
                        tensor = func(*list_args)
                        tensor = tensor.to(current_device, non_blocking=True)
                        if isinstance(tensor, DTensor):
                            tensor = tensor.full_tensor()
                    else:
                        list_args[6] = get_device_id()
                        tensor = func(*list_args)
                        if isinstance(tensor, DTensor):
                            tensor = tensor.full_tensor()
                    yield (name, tensor)

            rebuild_weights = rebuild_weights_generator()
            torch.cuda.synchronize()
            with self._maybe_tms_weights_region():
                loaded_params = self.model_runner.model.load_weights(weights=rebuild_weights)
            if blocking:
                # Ensure all operations are completed before returning
                torch.cuda.synchronize()
        except Exception as e:
            raise ValueError(f"Error in vLLMWorkerExtension.load_weights: {e}") from e
        return loaded_params

    def cuda_synchronize(self):
        """Synchronize the CUDA device."""
        try:
            torch.cuda.synchronize()
        except Exception as e:
            raise ValueError(f"Error in vLLMWorkerExtension.cuda_synchronize: {e}") from e
        return None

    def get_gpu_memory_snapshot(self) -> dict[str, int | str]:
        """Return process-local and device-wide CUDA memory from this vLLM worker."""
        try:
            device = torch.cuda.current_device()
            torch.cuda.synchronize(device)
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            properties = torch.cuda.get_device_properties(device)
            return {
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "node_id": self.get_node_id(),
                "rank": self.get_instance_local_rank(),
                "tp_rank": self.get_instance_local_tp_rank(),
                "device": device,
                "device_name": properties.name,
                "device_uuid": str(getattr(properties, "uuid", "unknown")),
                "torch_allocated_bytes": torch.cuda.memory_allocated(device),
                "torch_reserved_bytes": torch.cuda.memory_reserved(device),
                "device_used_bytes": total_bytes - free_bytes,
                "device_free_bytes": free_bytes,
                "device_total_bytes": total_bytes,
            }
        except Exception as e:
            raise ValueError(f"Error in vLLMWorkerExtension.get_gpu_memory_snapshot: {e}") from e

    def get_last_tms_sleep_wake_timing(self, operation: str) -> dict:
        """Return the last TMS timing captured inside this GPU worker."""
        if operation not in ("sleep", "wake"):
            raise ValueError(f"Unsupported TMS timing operation: {operation}")
        return {
            "hostname": socket.gethostname(),
            "rank": self.get_instance_local_rank(),
            "tp_rank": self.get_instance_local_tp_rank(),
            "pid": os.getpid(),
            "timing": getattr(self, f"_psrl_last_{operation}_timing", None),
        }

    def get_last_worker_stage_timing(self, stage: str) -> dict:
        """Return the latest timing for a worker-extension stage."""
        timings = getattr(self, "_psrl_worker_stage_timings", {})
        return {
            "hostname": socket.gethostname(),
            "rank": self.get_instance_local_rank(),
            "tp_rank": self.get_instance_local_tp_rank(),
            "pid": os.getpid(),
            "timing": timings.get(stage),
        }

    def get_weight_arena_info(self) -> dict:
        """Return JSON-safe arena and latest NIXL registration state."""
        handle = getattr(self.model_runner, "_psrl_weight_arena_handle", None)
        arena_cache = getattr(self, "_psrl_weight_arena_cpu_cache", None)
        cache_state = getattr(self, "_psrl_reward_weight_cache_state", None)
        cache_info = None
        if arena_cache is not None:
            cache_info = {
                "arena_count": len(arena_cache),
                "total_bytes": sum(tensor.numel() * tensor.element_size() for tensor in arena_cache),
                "pin_memory": all(tensor.is_pinned() for tensor in arena_cache),
            }
        registration_timing = None
        nixl_client = getattr(self, "nixl_storage_client", None)
        if nixl_client is not None:
            registration_timing = getattr(nixl_client, "_last_reregister_timing", None)
        if handle is None:
            return {
                "enabled": False,
                "hostname": socket.gethostname(),
                "node_id": self.get_node_id(),
                "rank": self.get_instance_local_rank(),
                "tp_rank": self.get_instance_local_tp_rank(),
                "pid": os.getpid(),
                "stats": None,
                "base_addresses": [],
                "reward_cpu_cache": cache_info,
                "reward_cpu_cache_state": cache_state,
                "nixl_registration": registration_timing,
            }
        handle.assert_virtual_addresses_unchanged()
        return {
            "enabled": True,
            "hostname": socket.gethostname(),
            "node_id": self.get_node_id(),
            "rank": self.get_instance_local_rank(),
            "tp_rank": self.get_instance_local_tp_rank(),
            "pid": os.getpid(),
            "stats": handle.stats.to_dict(),
            "base_addresses": list(handle.base_addresses),
            "reward_cpu_cache": cache_info,
            "reward_cpu_cache_state": cache_state,
            "nixl_registration": registration_timing,
        }

    def get_worker_stage_timing(self, stage: str) -> dict:
        """Return the latest extension timing with arena metadata."""
        return self.get_last_worker_stage_timing(stage)

    def patch_vllm_moe_model_weight_loader(self) -> None:
        """Patch the vLLM model weight loader for MoE models."""
        try:
            vllm_model = self.model_runner.model
            if isinstance(vllm_model, CUDAGraphWrapper):
                vllm_model = vllm_model.unwrap()
            patch_vllm_moe_model_weight_loader(vllm_model)
        except Exception as e:
            raise ValueError(f"Error in vLLMWorkerExtension.patch_vllm_moe_model_weight_loader: {e}") from e
        return None

    # ----------------------------- NIXL Related -----------------------------
    # Because the model is on another process since vllm V1, we must call the nixl methods via rpc
    def get_instance_local_rank(self):
        from vllm.distributed.parallel_state import get_world_group

        return get_world_group().rank

    def get_instance_local_tp_rank(self):
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

        return get_tensor_model_parallel_rank()

    def get_node_id(self) -> str:
        """Get the node id of the vllm worker."""
        if not hasattr(self, "node_id"):
            self.node_id = None

        if self.node_id is not None:
            return self.node_id
        self.node_id = ray.get_runtime_context().get_node_id()
        return self.node_id

    def init_nixl_client(
        self,
        nixl_config: DictConfig,
        nixl_interface_after_rpc: dict | NIXLInterface,
        instance_id: int,
        logging_path: str | None = None,
    ):
        # Reconstruct the nixl_interface (the RPC call serializes the nixl_interface to a dict)
        if isinstance(nixl_interface_after_rpc, dict):
            nixl_interface = NIXLInterface(port_scanner=nixl_interface_after_rpc["port_scanner"])
        else:
            nixl_interface = nixl_interface_after_rpc
        # NIXL attributes
        self.unified_state_dict = None
        self.unified_sharding_dict = None
        # Initialize the NIXL client
        self.nixl_storage_client = NIXLStorageClient(
            client_name=gen_client_name(instance_id, self.get_instance_local_rank()),
            server_name=NIXL_META_SERVER_NAME,
            use_gpu=True,
            client_type=NIXLClientType.PULL_SIDE,
            nixl_config=nixl_config,
            nixl_interface=nixl_interface,
            # client_group_id=instance_id,
            logging_path=logging_path,
        )
        psrl_logger.info(f"NIXL client initialized on port {self.nixl_storage_client.client_port}.")

    def nixl_convert_params(self, config: DictConfig):
        """Convert the model parameters to unified format.

        Args:
            config (DictConfig): Configuration object containing training settings.
        """
        from transformers import AutoConfig

        vllm_model = self.model_runner.get_model()
        model_config = AutoConfig.from_pretrained(
            copy_to_local(config.model.path),
            trust_remote_code=config.model.get("trust_remote_code", False),
        )
        parameter_mapping = create_parameter_mapping(type(vllm_model), model_config)
        self.unified_state_dict, self.local_sharding_dict = convert_vllm_inplace(
            parameter_mapping, vllm_model, tp_rank=self.get_instance_local_tp_rank()
        )

    def nixl_protocol(self, config: DictConfig, mode: str = "full"):
        """Run the NIXL server protocol.

        Args:
            config (DictConfig): Configuration object containing training settings.
            mode (str): Mode of registration, either 'meta' or 'full'.
                'meta' mode converts to meta tensors and skip registering their memory.
                'full' mode converts to full tensors.

            NOTE: ps storage may init with meta tensors, the register step would be different.
        """
        # Register the state dict and sharding dict to the NIXL client
        meta_only = mode == "meta"
        if self.unified_state_dict is None or self.local_sharding_dict is None:
            self.nixl_convert_params(config)
        psrl_logger.info("nixl client protocol step 1: connect_to_server")
        self.nixl_storage_client.connect_to_server()
        psrl_logger.info("nixl client protocol step 2: send_local_sharding")
        self.nixl_storage_client.send_local_sharding(self.local_sharding_dict)
        psrl_logger.info("nixl client protocol step 3: wait_for_server_sharding")
        unified_sharding_dict = self.nixl_storage_client.wait_for_server_sharding()
        # psrl_logger.info(f"unified_sharding_dict: {unified_sharding_dict}")
        psrl_logger.info("nixl client protocol step 4: register_local_tensors")
        self.nixl_storage_client.register_local_tensors(
            self.unified_state_dict, unified_sharding_dict, meta_only=meta_only
        )
        psrl_logger.info("nixl client protocol step 5: send_local_info")
        self.nixl_storage_client.send_local_info()
        psrl_logger.info("nixl client protocol step 6: wait_for_server_info")
        self.nixl_storage_client.wait_for_server_info()
        psrl_logger.info("nixl client protocol step 7: send_local_temp_mapping")
        self.nixl_storage_client.send_local_temp_mapping()
        psrl_logger.info("nixl client protocol step 8: wait_for_server_temp_mappings")
        self.nixl_storage_client.wait_for_server_temp_mappings()
        psrl_logger.info("nixl client protocol done.")
        self.unified_sharding_dict = unified_sharding_dict

    def nixl_register_after_wake_up(self):
        """Register the model parameters to NIXL after wake up from sleep.

        After sleep/wake_up, the physical memory backing the model weights
        has changed while virtual addresses remain the same. This method performs
        local re-registration:
        1. Reset nixl agent (clears UCX rcache)
        2. Re-registers memory with new physical pages (generates new rkeys)
        """
        total_start = time.perf_counter()
        sync_start = time.perf_counter()
        torch.cuda.synchronize()
        sync_elapsed_s = time.perf_counter() - sync_start
        arena_handle = getattr(self.model_runner, "_psrl_weight_arena_handle", None)
        arena_addresses = arena_handle.assert_virtual_addresses_unchanged() if arena_handle is not None else None
        # Reset nixl agent and reregister to handle physical memory changes
        register_start = time.perf_counter()
        self.nixl_storage_client.register_local_tensors(self.unified_state_dict, self.unified_sharding_dict)
        register_elapsed_s = time.perf_counter() - register_start
        elapsed_s = time.perf_counter() - total_start
        self._psrl_worker_stage_timings = getattr(self, "_psrl_worker_stage_timings", {})
        self._psrl_worker_stage_timings["nixl_register_after_wake_up"] = {
            "cuda_sync_elapsed_s": sync_elapsed_s,
            "register_elapsed_s": register_elapsed_s,
            "elapsed_s": elapsed_s,
            "arena_virtual_addresses": arena_addresses,
        }
        psrl_logger.warning(
            "[VLLM_SLEEP_WAKE_TIMING] scope=tp_worker operation=wake "
            "stage=nixl_register_after_wake_up host=%s rank=%s tp_rank=%s pid=%s "
            "cuda_sync_elapsed_s=%.6f register_elapsed_s=%.6f elapsed_s=%.6f "
            "arena_address_check=%s arena_count=%d",
            socket.gethostname(),
            self.get_instance_local_rank(),
            self.get_instance_local_tp_rank(),
            os.getpid(),
            sync_elapsed_s,
            register_elapsed_s,
            elapsed_s,
            "passed" if arena_addresses is not None else "disabled",
            len(arena_addresses or ()),
        )

    def nixl_deregister(self):
        """Deregister the model parameters from NIXL."""
        start = time.perf_counter()
        self.nixl_storage_client.deregister_local_tensors()
        elapsed_s = time.perf_counter() - start
        self._psrl_worker_stage_timings = getattr(self, "_psrl_worker_stage_timings", {})
        self._psrl_worker_stage_timings["nixl_deregister"] = {"elapsed_s": elapsed_s}
        psrl_logger.warning(
            "[VLLM_SLEEP_WAKE_TIMING] scope=tp_worker operation=sleep "
            "stage=nixl_deregister host=%s rank=%s tp_rank=%s pid=%s elapsed_s=%.6f",
            socket.gethostname(),
            self.get_instance_local_rank(),
            self.get_instance_local_tp_rank(),
            os.getpid(),
            elapsed_s,
        )

    def nixl_send_local_info_to(self, dst_agent_names: str | list[str]):
        """
        Send local NIXL info to the specified destination agents.
        """
        if isinstance(dst_agent_names, str):
            dst_agent_names = [dst_agent_names]
        self.nixl_storage_client.send_local_info_to(dst_agent_names)

    def nixl_update_local_info_to_ps(self, ps_worker_node_id_to_idxs: dict):
        """
        Update local NIXL info to the PS worker on the same node with this worker and PS manager.
        """
        node_id = self.get_node_id()
        dst_ps_worker_idx = ps_worker_node_id_to_idxs[node_id]
        dst_agent_names = [ps_agent_name(dst_ps_worker_idx), NIXL_META_SERVER_NAME]
        self.nixl_send_local_info_to(dst_agent_names)

    def nixl_wait_for_update_infos(self, info_num: int):
        """Wait for infos of updated clients for global synchronization.

        Args:
            info_num (int): Number of infos to wait for.
        """
        self.nixl_storage_client.wait_for_update_infos(info_num)

    def nixl_log_shard_info(self, label: str = "", max_elements: int = 8):
        """Debug log local NIXL shard info on this vLLM worker."""
        self.nixl_storage_client.log_shard_info(label=label, max_elements=max_elements)

    def nixl_pull_model_core(self, ps_nixl_agent_names, ps_nixl_gen_storage_client_names):
        """Pull the model parameters from PS workers via NIXL.

        Args:
            ps_nixl_agent_names (list[str]): List of PS NIXL agent names
            ps_nixl_train_storage_client_names (list[str]): List of PS NIXL train storage client names
        """
        if not hasattr(self, "pull_times"):
            self.pull_times = 0
        self.pull_times += 1
        total_start = time.perf_counter()
        wait_operations = []
        issue_reads_start = time.perf_counter()
        for key in self.unified_state_dict:
            for target_agent_name, target_client_name in zip(ps_nixl_agent_names, ps_nixl_gen_storage_client_names):
                shards_to_transfer = self.nixl_storage_client.client_read(
                    target_agent_name,
                    target_client_name,
                    key,
                    f"gen_pull_{self.pull_times}",
                )
                # shards_to_transfer = self.nixl_storage_client.client_read(
                #     target_agent_name, target_client_name, key, "gen_pull", merge_and_cache_xfer=False
                # )
                if len(shards_to_transfer) > 0:
                    wait_operations.append((key, target_client_name, shards_to_transfer))
        issue_reads_elapsed_s = time.perf_counter() - issue_reads_start
        # Generation cannot be overlapped with the NIXL pull, so we need to wait for all operations to complete
        wait_start = time.perf_counter()
        for key, target_client_name, shards_to_transfer in wait_operations:
            self.nixl_storage_client.wait(
                key,
                f"gen_pull_{self.pull_times}",
                "READ",
                target_client=target_client_name,
            )
            # self.nixl_storage_client.wait(key, "gen_pull", "READ", target_client=target_client_name)
        wait_elapsed_s = time.perf_counter() - wait_start
        finish_start = time.perf_counter()
        self.nixl_storage_client.merge_and_finish_cached_xfer()
        self.cuda_synchronize()
        # self.nixl_log_shard_info(label=f"AFTER_GEN_PULL_{self.pull_times}")
        self.nixl_storage_client.clear_intermediate_cached_data()
        finish_elapsed_s = time.perf_counter() - finish_start
        elapsed_s = time.perf_counter() - total_start
        self._psrl_worker_stage_timings = getattr(self, "_psrl_worker_stage_timings", {})
        self._psrl_worker_stage_timings["nixl_pull_model_core"] = {
            "issue_reads_elapsed_s": issue_reads_elapsed_s,
            "wait_elapsed_s": wait_elapsed_s,
            "finish_elapsed_s": finish_elapsed_s,
            "elapsed_s": elapsed_s,
        }
        psrl_logger.info(
            f"{self.nixl_storage_client}: NIXL pull model core done ({self.pull_times} times). "
            f"time: {elapsed_s}s"
        )

    def estimate_max_model_len(self):
        """Estimate the maximum model length that can fit in the available KV cache memory."""
        assert hasattr(self, "available_kv_cache_memory_bytes"), "available_kv_cache_memory_bytes must be set"
        assert hasattr(self, "vllm_config"), "vllm_config must be set"
        kv_cache_spec = self.get_kv_cache_spec()
        assert kv_cache_spec is not None, "kv_cache_spec must not be None"
        # It use the binary search to estimate the max model length
        actual_max_model_len = self.vllm_config.model_config.max_model_len
        # Set the max model length to the upper limit of the estimation
        self.vllm_config.model_config.max_model_len = self.vllm_config.additional_config.get(
            "max_model_len_used_in_estimation",
            self.vllm_config.model_config.max_model_len * 8192,
        )
        estimated_max_model_len = estimate_max_model_len(
            self.vllm_config, kv_cache_spec, self.available_kv_cache_memory_bytes
        )
        # Restore the actual max model length
        self.vllm_config.model_config.max_model_len = actual_max_model_len
        return estimated_max_model_len
