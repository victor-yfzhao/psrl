"""Unit tests for lightweight deployment-mode resolution and validation.

Verifies that each `psrl.deployment.mode` value resolves to the expected
`elastic_rm` flag combination and that backward-compat inference (mode=null)
preserves legacy behavior.
"""

import pytest
from omegaconf import OmegaConf
from psrl.utils.deployment_mode import (
    expand_ngpus_per_node,
    resolve_deployment_mode,
    validate_trainer_sleep_optimizer_offload,
)


def _make_config(
    mode,
    elastic_rm_enable=False,
    enable_trainer_pool=False,
    enable_policy=True,
    colocate=False,
    colocate_validate_and_train=False,
    optimizer_offload=False,
    strategy="fsdp2",
):
    cfg = OmegaConf.create(
        {
            "psrl": {
                "colocate": colocate,
                "colocate_validate_and_train": colocate_validate_and_train,
                "deployment": {
                    "mode": mode,
                    "trainer_pool_idle_rollout_instances": 0,
                    "trainer_pool_idle_rm_instances": 0,
                    "elastic_rm": {
                        "enable": elastic_rm_enable,
                        "shared_nnodes": 1,
                        "shared_ngpus_per_node": 8,
                        "enable_trainer_pool": enable_trainer_pool,
                        "enable_policy": enable_policy,
                    },
                },
            },
            "train_actor_rollout_ref": {
                "actor": {
                    "strategy": strategy,
                    "fsdp_config": {"optimizer_offload": optimizer_offload},
                    "megatron": {"optimizer_offload": optimizer_offload},
                }
            },
        }
    )
    return cfg


def _resolved(cfg):
    return resolve_deployment_mode(cfg), cfg.psrl.deployment.elastic_rm


def test_disaggregated():
    cfg = _make_config("disaggregated")
    mode, erm = _resolved(cfg)
    assert mode == "disaggregated"
    assert erm.enable is False
    assert erm.enable_trainer_pool is False


def test_elastic_rl():
    cfg = _make_config("elastic_rl", elastic_rm_enable=True, enable_policy=True)
    mode, erm = _resolved(cfg)
    assert mode == "elastic_rl"
    assert erm.enable is True


def test_trainer_pool_only():
    cfg = _make_config("trainer_pool_only")
    mode, erm = _resolved(cfg)
    assert mode == "trainer_pool_only"
    assert erm.enable is False
    assert erm.enable_trainer_pool is False
    assert erm.enable_policy is False


def test_colocated():
    cfg = _make_config("colocated")
    mode, erm = _resolved(cfg)
    assert mode == "colocated"
    assert erm.enable is True
    assert erm.enable_trainer_pool is True
    assert erm.enable_policy is False


def test_rollout_rm_colocated():
    cfg = _make_config("rollout_rm_colocated")
    mode, erm = _resolved(cfg)
    assert mode == "rollout_rm_colocated"
    assert erm.enable is True
    assert erm.enable_trainer_pool is False
    assert erm.enable_policy is False


def test_null_infers_elastic_rm():
    # mode=null + elastic_rm.enable=True -> elastic_rl (backward compat).
    cfg = _make_config(None, elastic_rm_enable=True)
    mode, erm = _resolved(cfg)
    assert mode == "elastic_rl"
    assert erm.enable is True


def test_null_infers_disaggregated():
    cfg = _make_config(None)
    mode, _ = _resolved(cfg)
    assert mode == "disaggregated"


def test_invalid_mode_raises():
    cfg = _make_config("bogus_mode")
    try:
        resolve_deployment_mode(cfg)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for invalid deployment mode")


def test_expand_ngpus_per_node_scalar():
    assert expand_ngpus_per_node(8, 2) == [8, 8]


def test_expand_ngpus_per_node_list_config():
    cfg = OmegaConf.create({"ngpus": [4, 8]})
    assert expand_ngpus_per_node(cfg.ngpus, 2) == [4, 8]


def test_expand_ngpus_per_node_rejects_wrong_length():
    cfg = OmegaConf.create({"ngpus": [4, 8]})
    try:
        expand_ngpus_per_node(cfg.ngpus, 3)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for mismatched per-node GPU list length")


def test_elastic_trainer_pool_requires_optimizer_offload():
    cfg = _make_config(
        "elastic_rl",
        elastic_rm_enable=True,
        enable_trainer_pool=True,
        optimizer_offload=False,
    )
    mode, _ = _resolved(cfg)

    with pytest.raises(ValueError, match=r"fsdp_config\.optimizer_offload=True"):
        validate_trainer_sleep_optimizer_offload(cfg, mode)


def test_elastic_trainer_pool_accepts_optimizer_offload():
    cfg = _make_config(
        "elastic_rl",
        elastic_rm_enable=True,
        enable_trainer_pool=True,
        optimizer_offload=True,
    )
    mode, _ = _resolved(cfg)

    validate_trainer_sleep_optimizer_offload(cfg, mode)


def test_trainer_pool_only_also_requires_optimizer_offload():
    cfg = _make_config("trainer_pool_only", optimizer_offload=False)
    mode, _ = _resolved(cfg)

    with pytest.raises(ValueError, match="trainer_pool_only"):
        validate_trainer_sleep_optimizer_offload(cfg, mode)


def test_colocated_validation_requires_optimizer_offload_without_elastic_pool():
    cfg = _make_config(
        "disaggregated",
        colocate_validate_and_train=True,
        optimizer_offload=False,
    )
    mode, _ = _resolved(cfg)

    with pytest.raises(ValueError, match="colocate_validate_and_train=True"):
        validate_trainer_sleep_optimizer_offload(cfg, mode)


def test_megatron_sleep_uses_megatron_optimizer_offload_setting():
    cfg = _make_config(
        "elastic_rl",
        elastic_rm_enable=True,
        enable_trainer_pool=True,
        optimizer_offload=False,
        strategy="megatron",
    )
    mode, _ = _resolved(cfg)

    with pytest.raises(ValueError, match=r"actor\.megatron\.optimizer_offload=True"):
        validate_trainer_sleep_optimizer_offload(cfg, mode)


if __name__ == "__main__":
    for fn in [
        test_disaggregated,
        test_elastic_rl,
        test_trainer_pool_only,
        test_colocated,
        test_rollout_rm_colocated,
        test_null_infers_elastic_rm,
        test_null_infers_disaggregated,
        test_invalid_mode_raises,
        test_expand_ngpus_per_node_scalar,
        test_expand_ngpus_per_node_list_config,
        test_expand_ngpus_per_node_rejects_wrong_length,
    ]:
        fn()
        print(f"PASS: {fn.__name__}")
    print("All deployment-mode resolution tests passed.")
