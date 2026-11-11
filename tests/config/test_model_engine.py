from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

pytestmark = pytest.mark.cpu_test

_CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "psrl/trainer/config")


@pytest.mark.parametrize(
    ("overrides", "strategy", "actor_class", "engine_key", "critic_strategy"),
    [
        ([], "fsdp2", "FSDPActorConfig", "fsdp_config", "fsdp"),
        (["model_engine=dp"], "fsdp2", "FSDPActorConfig", "fsdp_config", "fsdp"),
        (["model_engine=megatron"], "megatron", "McoreActorConfig", "megatron", "megatron"),
    ],
)
def test_model_engine_selects_matching_actor_config(overrides, strategy, actor_class, engine_key, critic_strategy):
    """
    Keep actor strategy and engine configuration consistent when selecting a backend.
    """
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        config = compose(config_name="ppo_trainer", overrides=overrides)

    actor = config.train_actor_rollout_ref.actor
    assert actor.strategy == strategy, "Actor strategy must match the selected backend."
    assert actor._target_ == f"verl.workers.config.{actor_class}", "Actor schema must match the selected backend."
    assert engine_key in actor, "Actor must contain its engine configuration."
    other_engine_key = "megatron" if engine_key == "fsdp_config" else "fsdp_config"
    assert other_engine_key not in actor, "Actor must not contain another backend's engine configuration."
    assert config.actor_rollout_ref.actor.strategy == strategy, "The veRL actor alias must use the selected backend."
    assert config.train_actor_rollout_ref.ref.strategy == strategy, (
        "Reference strategy must match the selected backend."
    )
    assert config.critic.strategy == critic_strategy, "Critic strategy must retain its backend default."
