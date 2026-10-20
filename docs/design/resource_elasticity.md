# Resource Elasticity

Resource elasticity shifts GPUs between policy rollout, generative reward model
(GenRM) inference, and training as their workloads change. This page explains the
runtime and control flow. For launch recipes, see {doc}`../examples/elastic`.
For configuration fields and defaults, see {doc}`resource_elasticity_configuration`.

## Why Do We Need Elasticity?

An RL step with generative rewards has three stages: generate policy responses,
judge them with a GenRM, and train on the resulting trajectories. Their resource
demands change throughout a step:

- Rollout and judging generate variable-length sequences, leaving a long tail
  after most requests finish.
- Reward demand arrives downstream of rollout, so the bottleneck can move between
  the two inference roles.
- Training needs a complete batch. Its GPUs can otherwise sit idle while rollout
  and reward evaluation finish.

Colocated stages take turns on the same GPUs, so another stage cannot fill the
tail of the current stage. Disaggregated stages can overlap, but their GPU split
stays fixed. Elastic deployment combines stage overlap with a changing allocation.

Resource elasticity lets a role release GPUs when another role can use them
more effectively, then reclaim them
when its own work becomes ready. Moving GPUs must include both **model parameters
and role context**, followed by redistributing requests onto the new allocation.

The design is described in
[Efficient Resource-Shiftable RL Post-training with Generative Rewards](https://arxiv.org/abs/2610.16866),
which calls the system **PivotRL**. The sections below connect its runtime,
planner, and shift-aware scheduling design to the PSRL implementation.

## Architecture

```{figure} /_static/img/resource_elasticity_overview.svg
:alt: Resource elasticity architecture with parameter storage, role shifting, schedulers, monitor, planner, and executor
:width: 100%
:align: center
:target: ../_static/img/resource_elasticity_overview.pdf

Resource-shiftable runtime, shift-aware schedulers, and the monitor-planner-executor loop.
```

Global Parameter Storage (GPS) keeps model parameters available across role
shifts. In PSRL, the {doc}`parameter_server` implements policy parameter storage
through NIXL. Frozen reward models restore checkpoint-backed CPU-cached weights
through their own worker path.

`main_ppo.py` builds two physical pools. `shared_rollout_pool` holds overlapping
rollout and reward placements. `train_pool` holds training and dedicated
validation replicas. With `enable_trainer_pool=true`, inference replicas may also
borrow training GPUs while the trainer sleeps.

| Role | Shift granularity | Ownership |
|------|-------------------|-----------|
| Policy rollout | Inference replica | Scaling policy and executor |
| GenRM inference | Inference replica | Scaling policy and executor |
| Training | Entire training pool | Training batch lifecycle |
| Validation | Dedicated replicas on training GPUs | Validation lifecycle |

`plan_replicas()` tiles each eligible pool using each role's `TP * PP * DP` world
size. A replica stays inside one pool. Placements are initialized once, so the
configured replica count includes sleeping replicas that cannot all run together.
`ClusterTopology` tracks their physical `(node_id, gpu_id)` sets and conflicts.

**Training keeps its world size and parallelism.** Elasticity changes which
pre-initialized role owns devices, without repartitioning a live trainer's
optimizer state or adding Ray nodes.

## Elastic Runtime: Sleep and Wakeup

The runtime decomposes a role shift into two operations: `sleep` releases the
donor's GPU storage, then `wakeup` restores the receiving role. In Python, the
worker methods spell the second operation `wake_up`. Actor processes, model
structure, and placement metadata survive the transition.

### API and Completion Contract

The inference control API is a coordinator command:

| Operation | Coordinator API | Completion means |
|-----------|-----------------|------------------|
| Sleep | `Command(type=CommandType.SLEEP, instance_ids=...)` | Routing is disabled for the selected replicas and their sleep operations have finished |
| Wakeup | `Command(type=CommandType.WAKE_UP, instance_ids=...)` | Storage and weights are restored, policy versions are published where needed, and routing is enabled |

`instance_ids` contains `(replica_id, dp_rank)` identifiers. The coordinator
expands them to whole replicas for memory transitions. The executor uses this
protocol after selecting placements and settling conflicting owners:

```python
from psrl.utils.server.command import Command, CommandType

await coordinator.exec_command.remote(
    Command(type=CommandType.SLEEP, instance_ids=instance_ids),
    blocking=True,
)
await coordinator.exec_command.remote(
    Command(type=CommandType.WAKE_UP, instance_ids=instance_ids),
    blocking=True,
)
```

Here `coordinator` is a scoped `CoordinatorClient` on the shared `GlobalCoordinator` for one
role/model, and `instance_ids` selects its registered instances. This is the
runtime protocol used by the executor. Direct worker calls do not perform the
executor's GPU ownership checks.

The worker APIs beneath this protocol are:

| Role | Sleep path | Wakeup path |
|------|------------|-------------|
| Policy rollout | `server.nixl_sleep(level=2)` | `server.nixl_wake_up()`, then `pull_model_for_sync(ps_model_version)` |
| Frozen GenRM | `server.sleep(level=2)` | `server.wake_up()`, including restoration from its CPU weight cache |
| Training | `actor_wg.execute_all_async("nixl_sleep", "full")` | `nixl_wake_up` on all workers, refresh client metadata, then `pull_model` |

Level-2 inference sleep releases GPU weight storage. A policy server deregisters
NIXL tensors **before their physical pages are released**. Its wakeup reallocates
and registers storage, then the coordinator pulls committed policy weights and
publishes the actual loaded version before enabling dispatch.

For a GenRM, wakeup restores weight storage first, loads CPU-cached weights, then
restores KV/cache allocations and supported graph allocations before reopening
generation admission. Frozen reward weights do not use policy version updates.

### What Happens to Role Context?

| State | Sleep | Wakeup |
|-------|-------|--------|
| Policy parameters | Release GPU storage after NIXL deregistration | Allocate/register storage and pull committed parameters |
| Frozen reward parameters | Release GPU storage, retain CPU weight cache | Load the cached checkpoint weights |
| KV cache and temporary workspaces | Release disposable GPU allocations | Reallocate and recompute request state as needed |
| Optimizer state | Offload to CPU | Restore according to the training backend's offload configuration |
| CUDA graph allocations | Handle through the selected backend and TMS graph support | Restore or reconstruct through that backend's wake path |
| Processes, model layout, and communication metadata | Retain the role's initialized structure | Reuse the initialized role |

The paper's Figure 6 motivates separating disposable state from state that must
survive a shift. In PSRL, TMS manages supported allocations through pause/resume
and tagged regions. **TMS alone does not allocate GPUs to roles**. The executor
and coordinators establish ownership. Keep optimizer offload enabled whenever
training can sleep. See {doc}`resource_elasticity_configuration` for the backend
and TMS settings.

### Inference Role Shift

```{mermaid}
:config: {"sequence": {"wrap": false, "width": 100, "height": 35, "actorMargin": 20, "messageMargin": 12, "boxMargin": 5, "diagramMarginY": 5, "mirrorActors": false}}

sequenceDiagram
    participant E as Executor
    participant D as Donor
    participant W as Worker
    participant R as Receiver
    E->>D: SLEEP(instance_ids)
    D->>D: Pause dispatch
    D->>W: Deregister (policy),<br/>abort and release storage
    W-->>D: Asleep
    D-->>E: SLEEP confirmed
    E->>R: WAKE_UP(instance_ids)
    R->>W: Allocate and register,<br/>restore weights
    W-->>R: Weights ready
    R->>R: Publish policy version<br/>and enable routing
    R-->>E: WAKE_UP confirmed
    E->>R: Apply planned migrations
```

Replicas within a lifecycle batch may transition concurrently. Conflicting donor
and receiver batches are ordered: donor sleep completes before receiver wakeup.
Coordinators reserve the affected replicas against concurrent lifecycle, weight
synchronization, and migration operations. Failed restores remain outside routing.

### Training Pool Release and Reclaim

The full-batch training strategy uses a separate ownership boundary:

1. `release_trainer_pool_for_wait(buffer_id)` checks batch readiness for
   `trainer_ready_grace_s`. If the batch is ready, an awake trainer keeps its
   storage. Otherwise, all training workers sleep and the pool becomes borrowable.
2. While training waits, the policy may wake inference replicas on that pool.
3. When the batch is ready, `acquire_trainer_pool()` blocks new borrowing, waits
   for the current scaling decision, and sleeps every inference borrower.
4. The executor wakes trainer storage, refreshes NIXL client descriptors across
   participants, and pulls policy parameters. Training starts after these finish.

Validation uses the same exclusive pool boundary. It reclaims inference
borrowers, sleeps the trainer, then wakes the dedicated validation replicas.
Initialization also orders overlapping roles through sleep before another role
allocates its storage.

## Monitor, Planner, and Executor

`ElasticExecutor` runs the monitor and execution loops. Its scaling policy
implements the planner.

| Component | Input | Responsibility |
|-----------|-------|----------------|
| Monitor | Engine snapshots, router queues, placements, trainer readiness | Build a consistent view of load and available devices |
| Planner | Instance signals and, when enabled, immutable request snapshots | Return a `ScalingDecision` containing exact wake/sleep targets and optional request migrations |
| Executor | Accepted decision and physical topology | Sleep conflicting donors, wake receivers, apply migrations, and update ownership |

The monitor polls coordinators every `monitor_interval_ms`. Observations include
KV utilization, running/waiting counts, token occupancy, and throughput. The
request-level path also captures request lengths, queue order, model versions,
and routing constraints. Failed snapshot collection skips that planning cycle.
Pool and instance epochs prevent a plan from using observations collected across
an ownership transition.

The planner is selected with `scaling_policy_variant`. The two main
implementations, `rule-based` and `throughput-optimal`, are explained in
{doc}`resource_elasticity_scaling_policies`. Training reclamation remains a
training-loop operation, outside the inference throughput objective.

The executor keeps **one policy decision in flight** until all its actions
settle. A command acknowledgement timeout does not cancel sleep or wakeup. The
executor follows the command's completion record and retains ownership while
waiting. Runtime failures surface to later executor operations.

Implementation entry points are `psrl/workers/elastic/executor.py`,
`cluster_topology.py`, and `scaling_policy/factory.py`. Coordinator lifecycle
implementations live in `psrl/workers/global_coordination/commands.py`
and `psrl/workers/global_coordination/lifecycle.py`.

## Migration

Changing the replica count does not automatically redistribute existing work.
PSRL uses **abort and redispatch** to move requests through the shared routing
queue. The destination reconstructs execution from the request and generated
prefix. A device-to-device copy of the source KV cache is not required for this
path. See {doc}`flexible_rollout` for partial rollout and {doc}`kv_cache` for
optional cache reuse.

```{mermaid}
flowchart TB
    A["Source replica<br/>Running request and generated prefix"]
    B["Abort / interrupted response"]
    Q["Shared routing queue<br/>Preserve continuation state"]
    D["Eligible destination<br/>Recompute prefix and continue"]
    A --> B --> Q --> D
    DOWN["Scale down<br/>Remove source from routing"] --> B
    UP["Scale up<br/>Restore receiver, then rebalance"] --> B
```

**Scale down** removes the donor from routing before the engine sleeps in abort
mode. Interrupted policy requests return through the partial-rollout path and
are redispatched to eligible awake replicas.

**Scale up** makes new capacity eligible for queued requests. With
`rebalance_after_scale_up=true`, the throughput policy can also redistribute
existing requests. The request-level simulator considers short movable requests
from high-throughput donors and retains moves that improve aggregate throughput.
The selected source, destination, and request IDs are attached to the decision.

For each explicit migration, the coordinator:

1. Reserves source and destination replicas and resolves the external request ID.
2. Registers the destination with the gateway's migration endpoint.
3. Aborts the source request so its continuation returns to routing.
4. Waits for destination dispatch or request completion, then finishes the
   migration record.

`exclusive_rebalance_migration_queue=true` pauses ordinary dispatch while
selected migrations are applied. Model-version eligibility still constrains
destinations. Requests that have already finished need no move.

**The live snapshot adapter marks existing reward requests as non-movable.**
The paper describes a general shift-aware scheduler, but the current request-level
GenRM path does not provide the same active-request migration contract as policy
rollout. Added GenRM capacity can accept queued and future requests.

Snapshot construction is in `psrl/workers/elastic/request_snapshot.py`.
Simulation is in `scaling_policy/evaluation/request_level_candidate_evaluator.py`.
Migration execution is in the inference coordinator's command service.

```{toctree}
:maxdepth: 1
:hidden:

resource_elasticity_scaling_policies
resource_elasticity_configuration
```
