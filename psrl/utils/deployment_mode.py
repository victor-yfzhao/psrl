"""Standalone deployment-mode resolution (no heavy imports).

`resolve_deployment_mode` maps the unified `psrl.deployment.mode` config entry
to the concrete `elastic_rm` flags that the rest of the codebase reads. It must
run BEFORE any code that branches on `elastic_rm.enable` / `enable_trainer_pool`
/ `enable_policy` — in particular before `main_ppo.TaskRunner.init_resource_pool_mgr`
(which lays out resource pools) and before `PSRL_RayPPOTrainer.__init__` derives
its mode flags.

It is intentionally free of ray / torch / nixl imports so it can be called from
`main_ppo` at the very top of `TaskRunner.run` without triggering implicit
`ray.init()`.
"""

from omegaconf import ListConfig, open_dict

VALID_DEPLOYMENT_MODES = (
    "disaggregated",
    "colocated",
    "rollout_rm_colocated",
    "trainer_pool_only",
    "elastic_rl",
)


def expand_ngpus_per_node(ngpus_per_node, nnodes, field_name: str = "ngpus_per_node") -> list[int]:
    """Normalize scalar or per-node GPU specs to one integer per node.

    `ngpus_per_node` may be a scalar, a Python sequence, or an OmegaConf
    ListConfig. Scalars are repeated `nnodes` times; sequences must already
    match `nnodes`.
    """
    nnodes = int(nnodes)
    if nnodes < 0:
        raise ValueError(f"nnodes must be >= 0, got {nnodes}.")

    if isinstance(ngpus_per_node, (list, tuple, ListConfig)):
        spec = [int(v) for v in ngpus_per_node]
        if len(spec) != nnodes:
            raise ValueError(
                f"{field_name} has {len(spec)} entries but shared_nnodes={nnodes}; "
                "provide one GPU count per shared node."
            )
    else:
        spec = [int(ngpus_per_node)] * nnodes

    if any(v < 0 for v in spec):
        raise ValueError(f"{field_name} values must be >= 0, got {spec}.")
    return spec


def resolve_deployment_mode(config) -> str:
    """Resolve `psrl.deployment.mode` into concrete elastic_rm flags.

    Mutates `config` in place so downstream code sees a consistent flag set.
    When `mode` is null, infer from existing flags for backward compatibility.
    Returns the resolved mode string. Idempotent: safe to call multiple times.
    """
    deployment = config.psrl.deployment
    mode = deployment.get("mode", None)
    if mode is None:
        # Backward-compat inference: prefer existing elastic_rm.enable, then
        # colocate, falling back to disaggregated.
        if deployment.elastic_rm.get("enable", False):
            mode = "elastic_rl"
        elif config.psrl.get("colocate", False):
            # Legacy sync colocate path; keep disaggregated semantics for pool
            # layout (the sync path does not use elastic pools).
            mode = "disaggregated"
        else:
            mode = "disaggregated"
    mode = str(mode)
    if mode not in VALID_DEPLOYMENT_MODES:
        raise ValueError(
            f"Invalid psrl.deployment.mode={mode!r}; expected one of {VALID_DEPLOYMENT_MODES}."
        )

    with open_dict(config):
        config.psrl.deployment.mode = mode
        elastic_rm = config.psrl.deployment.elastic_rm
        if mode == "disaggregated":
            elastic_rm.enable = False
            elastic_rm.enable_trainer_pool = False
        elif mode == "elastic_rl":
            elastic_rm.enable = True
            # enable_trainer_pool and enable_policy stay as user-configured.
        elif mode == "trainer_pool_only":
            # Disaggregated baseline (rollout/rm on independent pools, always
            # awake) + a fixed number of extra rollout/rm replicas on train_pool
            # that are woken while the trainer is idle. No elastic auto-scaling,
            # no shared_rollout_pool. NIXL is required for actor sleep/wake.
            elastic_rm.enable = False
            elastic_rm.enable_trainer_pool = False
            elastic_rm.enable_policy = False
        elif mode == "colocated":
            # All three roles share train_pool (colocated), time-multiplexed per
            # training step. Reuse elastic SubRayResourcePool / bundle-mapping /
            # coordinator sleep/wake + trainer-pool actor NIXL sleep/wake infra,
            # but disable auto-scaling; SleepWakeOrchestrator drives phases.
            elastic_rm.enable = True
            elastic_rm.enable_trainer_pool = True
            elastic_rm.enable_policy = False
        elif mode == "rollout_rm_colocated":
            # rollout+rm share shared_rollout_pool (time-multiplexed per buffer);
            # trainer on a separate train_pool, pipelined via staleness.
            elastic_rm.enable = True
            elastic_rm.enable_trainer_pool = False
            elastic_rm.enable_policy = False
    return mode
