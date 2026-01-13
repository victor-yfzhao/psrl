# Modified from verl/experimental/reward/reward_loop/registry.py
import asyncio
import importlib
import logging
import os
from collections.abc import Callable
from functools import partial
import sys
from typing import Any

from omegaconf import DictConfig
from verl.trainer.ppo.reward import get_custom_reward_fn

from psrl.utils.reward_score import default_compute_score_async
from psrl.workers.reward.reward_loop.base import RewardLoopManagerBase
from psrl.workers.reward.gen_reward_function import get_gen_reward_function_cls
from psrl.workers.reward.reward_model import PSRL_RewardModelManager

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

__all__ = ["register", "get_reward_loop_manager_cls", "load_reward_loop_manager"]

REWARD_LOOP_MANAGER_REGISTRY: dict[str, type[RewardLoopManagerBase]] = {}


def register(
    name: str,
) -> Callable[[type[RewardLoopManagerBase]], type[RewardLoopManagerBase]]:
    """Decorator to register a reward loop manager class with a given name.

    Args:
        name: `(str)`
            The name of the reward loop manager.
    """

    def decorator(cls: type[RewardLoopManagerBase]) -> type[RewardLoopManagerBase]:
        if name in REWARD_LOOP_MANAGER_REGISTRY and REWARD_LOOP_MANAGER_REGISTRY[name] != cls:
            raise ValueError(
                f"reward loop manager {name} has already been registered: "
                f"{REWARD_LOOP_MANAGER_REGISTRY[name]} vs {cls}"
            )
        REWARD_LOOP_MANAGER_REGISTRY[name] = cls
        return cls

    return decorator


def get_reward_loop_manager_cls(name: str) -> type[RewardLoopManagerBase]:
    """Get the reward loop manager class with a given name.

    Args:
        name: `(str)`
            The name of the reward loop manager.

    Returns:
        `(type)`: The reward loop manager class.
    """
    if name not in REWARD_LOOP_MANAGER_REGISTRY:
        raise ValueError(f"Unknown reward loop manager: {name}")
    return REWARD_LOOP_MANAGER_REGISTRY[name]


def load_reward_loop_manager(
    reward_model_config: DictConfig,
    input_tokenizer: Any,
    reward_loop_type: str,
    reward_fn: str | dict[str, str],
    reward_model_manager: PSRL_RewardModelManager = None,
    **reward_kwargs: Any,
) -> RewardLoopManagerBase:
    """Load the reward loop manager based on the configuration.

    Args:
        reward_model_config: `(DictConfig)`
            The configuration for the reward model.
        reward_loop_type: `(str)`
            The type of the reward loop manager.
        reward_fn: `(str | dict[str, str])`
            The name of the reward function.
            Or, a customized reward function, like `{name: ..., path: ...}`
        input_tokenizer: `(Any)`
            The tokenizer for the input.
        reward_model_manager: `(PSRL_RewardModelManager)`
            The reward model manager.
        **reward_kwargs: `(Any)`
            Additional keyword arguments for the reward loop manager.
    Returns:
        `(RewardLoopManagerBase)`: The reward loop manager instance.
    """
    reward_loop_manager_name = reward_loop_type
    reward_loop_manager_cls = get_reward_loop_manager_cls(reward_loop_manager_name)

    # Try to get a custom reward function based on the configuration
    # user defined reward manager can be registered in custom_reward_fn
    if isinstance(reward_fn, dict):
        compute_score = get_custom_reward_fn(reward_fn)
        final_compute_score = compute_score
        if final_compute_score is None:
            sandbox_config = reward_kwargs.get("sandbox_fusion", None)
            sandbox_url = sandbox_config.get("url") if sandbox_config else None
            memory_limit_mb = sandbox_config.get("memory_limit_mb", 1024)
            if sandbox_url:
                # Create an asyncio.Semaphore to control concurrent access to the sandbox
                # Note: asyncio.Semaphore must be created in the same event loop where it will be used
                # Therefore, we pass max_concurrent as a parameter and create the semaphore later
                max_concurrent = reward_kwargs.get("max_concurrent", 64)
                _concurrent_semaphore = asyncio.Semaphore(max_concurrent)
                final_compute_score = partial(
                    default_compute_score_async,
                    sandbox_fusion_url=sandbox_url,
                    concurrent_semaphore=_concurrent_semaphore,
                    memory_limit_mb=memory_limit_mb,
                )
    else:
        final_compute_score = default_compute_score_async        

    if reward_loop_manager_name == "gen":
        # Get gen_reward_function from config or reward_kwargs
        gen_reward_function_name = reward_fn
        gen_reward_function_cls = get_gen_reward_function_cls(gen_reward_function_name)
        return reward_loop_manager_cls(
            reward_model_config,
            input_tokenizer,
            reward_model_manager=reward_model_manager,
            reward_function=gen_reward_function_cls(),
            **reward_kwargs,
        )

    return reward_loop_manager_cls(
        reward_model_config,
        input_tokenizer,
        final_compute_score,
        **reward_kwargs,
    )

# Modified from verl/trainer/ppo/reward.py
def get_custom_reward_fn(reward_fn: dict[str, Any]) -> Callable[[], Any]:
    """Load and return a custom reward function from external file.

    Dynamically imports a reward function from a specified file path and wraps
    it with additional keyword arguments from the configuration.

    Args:
        reward_fn: `(dict[str, Any])`
            A customized reward function, like `{name: ..., path: ...}`

    Returns:
        callable or None: Wrapped reward function with merged kwargs, or None
                         if no custom reward function is configured.

    Raises:
        FileNotFoundError: If the specified reward function file doesn't exist.
        RuntimeError: If there's an error loading the module from file.
        AttributeError: If the specified function name isn't found in the module.
    """

    file_path = reward_fn.get("path")
    function_name = reward_fn.get("name")

    module = sys.modules.get("custom_module", None)
    if module is None:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

        spec = importlib.util.spec_from_file_location("custom_module", file_path)
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules["custom_module"] = module
            assert spec.loader is not None
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Error loading module from '{file_path}': {e}") from e

    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{module.__file__}'.")

    print(f"using customized reward function '{function_name}' from '{module.__file__}'")
    raw_fn = getattr(module, function_name)

    reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))

    if not inspect.iscoroutinefunction(raw_fn):
        return partial(_call_with_kwargs, raw_fn, reward_kwargs)
    else:
        return partial(_call_with_kwargs_async, raw_fn, reward_kwargs)