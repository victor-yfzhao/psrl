import asyncio
import logging
import os
import time
import warnings

import numpy as np
import ray

from psrl.utils.elastic_rm.diagnostics import log_elastic_rm_backlog_diag
from psrl.utils.logger import (
    DualOutputHandler,
    EventType,
    deprecated,
    log_dual_events,
)
from psrl.utils.server.command import Command, CommandExtension, CommandType
from psrl.workers.gen.stats_collector import EngineStats

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "INFO"))


@ray.remote
class RewardModelCoordinator(CommandExtension):
    def __init__(
        self,
        config,
        rm_config,
        status_queues,
    ):
        """
        Initialize the RewardModelCoordinator.
        Coordinates and manages reward model instances for PSRL.

        This class handles:
        - Registering and tracking reward model instances
        - Managing model version synchronization across instances
        - Handling command execution (abort, sync)
        - Collecting and distributing engine status information
        - Coordinating interruption and resumption of generation tasks

        Args:
            config: Configuration object containing PSRL settings
            status_queues: Queues for receiving status updates from different reward model instances
        """
        super().__init__()

        self.config = config
        self.rm_config = rm_config
        self.reward_model_name = rm_config.reward_model_name

        # will be set by the reward model manager later
        self.reward_model_wg_list = None
        self.reward_model_wg_size = None
        self.reward_model_router = None

        # Stats collection
        self.status_queues = status_queues
        assert len(self.status_queues) == self.rm_config.num_replicas, (
            "The number of status queues must be the same as the number of rollout instances."
        )

        # Background event handler
        self.running_loop = None
        self.command_handler_task = None
        self._active_command_id: int | None = None
        self.process_status_queue_tasks = []
        self.broadcast_status_to_router_task = None
        self.stop_command_handler = False
        self.stop_process_status_queue = [False] * self.rm_config.num_replicas
        self.stop_broadcast_status_to_router = False

        # Asyncio event loop order control
        self._is_init_model = asyncio.Event()

        # Engine status tracking
        self.instance_to_engine_status: dict[int, EngineStats] = {}  # Track the latest engine stats of each instance

        # Build logger
        self.log_prefix = "RewardModelCoordinator"
        psrl_logger.addHandler(DualOutputHandler(self.config.psrl.logging_path, self.log_prefix))

    def world_size(self):
        return sum([reward_model_wg.world_size for reward_model_wg in self.reward_model_wg_list])

    def set_reward_model_wg_list(self, reward_model_wg_list):
        self.reward_model_wg_list = reward_model_wg_list
        self.reward_model_wg_size = len(reward_model_wg_list)
        assert self.reward_model_wg_size == self.rm_config.num_replicas, (
            "The number of reward model worker groups must be the same as the number of reward model instances."
        )
        psrl_logger.info(f"Set reward model worker group list for {self.reward_model_name}, size: {self.reward_model_wg_size}")

    def set_reward_model_router(self, reward_model_router):
        self.reward_model_router = reward_model_router
        psrl_logger.info("Set reward model router for %s", self.reward_model_name)

    async def init_model(self):
        assert (
            self.reward_model_wg_list is not None
        ), "Reward model worker group list must be set before initializing the model."
        init_futures = []
        for i in range(self.reward_model_wg_size):
            init_futures.extend(self.reward_model_wg_list[i].execute_all_async("init_model", "empty"))
        await asyncio.gather(*init_futures)
        psrl_logger.info(f"Reward model {self.reward_model_name} initialized.")

        if self.config.psrl.deployment.elastic_rm.enable:
            # Only rank-0 workers own vLLM AsyncLLM when TP/PP > 1 (see PSRL_vLLMRollout).
            sleep_futures = []
            for i in range(self.reward_model_wg_size):
                sleep_futures.append(self.reward_model_wg_list[i].execute_rank_zero_async("sleep"))
            await asyncio.gather(*sleep_futures)
            psrl_logger.info(f"Reward model {self.reward_model_name} sleeping.")

        self._is_init_model.set()
    
    async def start_busy_loop(self):
        """
        Start the background event loops for command handling and status synchronization.

        This method:
        1. Starts a background task for handling commands (abort, sync, etc.).
        2. Optionally starts tasks for processing status queues of each rollout instance.
        3. Starts a task to broadcast the engine status to the agent loop workers (i.e., router).
        """
        await self._is_init_model.wait()

        if self.command_handler_task is not None and not self.command_handler_task.done():
            return

        # Start the background tasks
        self.running_loop = asyncio.get_running_loop()
        self.command_handler_task = self.running_loop.create_task(self._command_handler_loop())
        self.command_handler_task.add_done_callback(self._on_command_handler_done)

        # Start the status collection tasks
        if self.config.psrl.status_collection.enable:
            for instance_id in range(self.rm_config.num_replicas):
                self.process_status_queue_tasks.append(
                    self.running_loop.create_task(self._process_status_queue(instance_id))
                )
                self.process_status_queue_tasks[instance_id].add_done_callback(
                    lambda f: f.result()
                )  # To avoid silent error in async tasks

            # Broadcast collected engine status to the RM router so cost-model
            # route strategies (throughput_optimal*) can estimate per-instance
            # load/throughput. The task tolerates the router being set late
            # (manager sets it after start_busy_loop returns).
            self.broadcast_status_to_router_task = self.running_loop.create_task(
                self._broadcast_status_to_router()
            )
            self.broadcast_status_to_router_task.add_done_callback(lambda f: f.result())

        # # Start the model synchronization and rollout migration loop
        # if self.config.psrl.sync_and_mig_strategy.method == "greedy":
        #     self.sync_task = self.running_loop.create_task(self._greedy_sync_and_migrate_loop())
        #     self.sync_task.add_done_callback(lambda f: f.result())  # To avoid silent error in async tasks
        # elif self.config.psrl.sync_and_mig_strategy.method == "status_based":
        #     assert self.config.psrl.status_collection.enable, (
        #         "Status-based sync strategy is only supported when status collection is enabled"
        #     )
        #     self.sync_task = self.running_loop.create_task(self._status_based_sync_and_migrate_loop())
        #     self.sync_task.add_done_callback(lambda f: f.result())  # To avoid silent error in async tasks
        # else:
        #     raise NotImplementedError(
        #         f"Sync strategy {self.config.psrl.sync_and_mig_strategy.method} is not supported"
        #     )

    async def stop_busy_loop(self):
        """
        Stop all background tasks and clean up resources.

        This method gracefully shuts down:
        - Command handler task
        - Engine status sync task
        """
        if self.command_handler_task is None or self.command_handler_task.done():
            return

        # Stop the background tasks
        self.stop_command_handler = True
        self.stop_process_status_queue = [True] * self.rm_config.num_replicas
        self.stop_broadcast_status_to_router = True

        tasks_to_wait = [self.command_handler_task]
        if self.process_status_queue_tasks:
            tasks_to_wait.extend(self.process_status_queue_tasks)
        if self.broadcast_status_to_router_task is not None:
            tasks_to_wait.append(self.broadcast_status_to_router_task)

        # Wait for tasks to finish with timeout
        await asyncio.gather(*tasks_to_wait, return_exceptions=True)

    async def _command_handler_loop(self):
        """
        Background loop for processing commands from the command queue.

        This method continuously processes different types of commands:
        - ABORT: Interrupt specific requests on instances
        - SLEEP/WAKE_UP:

        The loop runs until stop_command_handler is set to True.
        """
        while not self.stop_command_handler:
            # Command processing
            if not self.command_queue.empty():
                command = self.command_queue.get_nowait()

                assert isinstance(command, Command), f"Expected Command type, got {type(command)}"

                # Unpack command attributes
                command_type = command.type
                command_id = command.get_kwargs()["id"]
                self._active_command_id = command_id
                self._start_command(command_id)
                command_args = command.get_args()
                psrl_logger.info(
                    "Receive command: type = %s, kwargs = %s, args_keys = %s",
                    command_type,
                    command.get_kwargs(),
                    list(command_args.keys()),
                )

                result = None
                # Process the command based on its type
                if command_type == CommandType.ABORT:
                    instance_to_uids = command_args.get("instance_to_uids", None)
                    instance_ids = command_args.get("instance_ids", None)
                    request_migrations = command_args.get("request_migrations", None)
                    migration_context = command_args.get("migration_context")
                    if instance_to_uids is None and instance_ids is None and request_migrations is None:
                        raise ValueError(
                            "ABORT command must contain 'instance_to_uids', 'instance_ids', "
                            "or 'request_migrations' in args."
                        )

                    if request_migrations is not None:
                        if self.reward_model_router is None:
                            instance_to_uids = {}
                        else:
                            prepared = await self.reward_model_router.prepare_request_migrations.remote(
                                request_migrations
                            )
                            instance_to_uids = prepared.get("instance_to_uids", {})
                            psrl_logger.info(
                                "Prepared RM request migrations before ABORT: %s",
                                prepared,
                            )

                    psrl_logger.info(
                        "Received ABORT command with instance_to_uids (count=%s) and instance_ids=%s; "
                        "completing immediately (fire-and-forget) to avoid blocking WAKE_UP/SLEEP commands.",
                        sum(len(v) for v in (instance_to_uids or {}).values()),
                        instance_ids,
                    )
                    futures = []

                    if migration_context and instance_to_uids is not None and self.reward_model_router is not None:
                        await self.reward_model_router.mark_migration_requests.remote(
                            instance_to_uids,
                            migration_context,
                        )

                    if instance_to_uids is not None:
                        for instance_id, uids in instance_to_uids.items():
                            if not uids:
                                continue
                            if not isinstance(uids, (list, set)):
                                uids = [uids]
                            # Normalize request IDs to int to avoid string/int mismatch
                            # between scheduler stats and worker-side active task map keys.
                            abort_requests: set[str] = set()
                            for uid in uids:
                                if uid is None:
                                    psrl_logger.warning(
                                        "Skip empty uid in ABORT command for instance %s: uid=%r",
                                        instance_id,
                                        uid,
                                    )
                                    continue
                                abort_requests.add(str(uid))
                            if not abort_requests:
                                continue
                            futures.append(
                                self.reward_model_wg_list[instance_id].execute_rank_zero_async(
                                    "interrupt_requests", abort_requests
                                )
                            )
                    if instance_ids is not None:
                        for instance_id in instance_ids:
                            futures.append(
                                self.reward_model_wg_list[instance_id].execute_rank_zero_async(
                                    "interrupt_requests", None
                                )
                            )

                    if not futures:
                        interrupted_request_num = 0
                    else:
                        interrupted_request_nums = await asyncio.gather(*futures)
                        interrupted_request_num = np.sum(interrupted_request_nums)

                    result = interrupted_request_num
                    psrl_logger.info(f"Received ABORT command, interrupted {interrupted_request_num} requests")
                    # Post process the command result
                    self._complete_command(command_id, result)
                elif command_type == CommandType.SLEEP:
                    instance_ids = command_args.get("instance_ids", None)
                    if instance_ids is None:
                        raise ValueError("SLEEP command must contain 'instance_ids' in args.")
                    if not isinstance(instance_ids, list):
                        instance_ids = [instance_ids]

                    if -1 in instance_ids:
                        instance_ids = list(range(self.reward_model_wg_size))

                    overhead_started_s = time.monotonic()
                    pause_s = 0.0
                    interrupt_s = 0.0
                    engine_sleep_s = 0.0
                    success = False
                    try:
                        if self.reward_model_router is not None:
                            stage_started_s = time.monotonic()
                            await self.reward_model_router.pause_instances.remote(instance_ids)
                            pause_s = time.monotonic() - stage_started_s
                        stage_started_s = time.monotonic()
                        await asyncio.gather(
                            *[
                                self.reward_model_wg_list[instance_id].execute_rank_zero_async(
                                    "interrupt_generation"
                                )
                                for instance_id in instance_ids
                            ]
                        )
                        interrupt_s = time.monotonic() - stage_started_s
                        stage_started_s = time.monotonic()
                        await asyncio.gather(
                            *[
                                self.reward_model_wg_list[instance_id].execute_rank_zero_async("sleep")
                                for instance_id in instance_ids
                            ]
                        )
                        engine_sleep_s = time.monotonic() - stage_started_s
                    except Exception:
                        psrl_logger.exception("SLEEP command failed for RM instances %s", instance_ids)
                        if self.reward_model_router is not None:
                            await self.reward_model_router.resume_instances.remote(instance_ids)
                        self._complete_command(command_id, False)
                    else:
                        success = True
                        psrl_logger.info("SLEEP command for instances %s completed.", instance_ids)
                        self._complete_command(command_id, True)
                    finally:
                        psrl_logger.info(
                            "[ELASTIC_OVERHEAD] operation=sleep role=RewardModel instances=%s success=%s "
                            "router_pause_s=%.6f interrupt_s=%.6f engine_sleep_s=%.6f total_s=%.6f",
                            instance_ids,
                            success,
                            pause_s,
                            interrupt_s,
                            engine_sleep_s,
                            time.monotonic() - overhead_started_s,
                        )
                
                elif command_type == CommandType.WAKE_UP:
                    instance_ids = command_args.get("instance_ids", None)
                    if instance_ids is None:
                        raise ValueError("WAKE_UP command must contain 'instance_ids' in args.")
                    if not isinstance(instance_ids, list):
                        instance_ids = [instance_ids]

                    if -1 in instance_ids:
                        instance_ids = list(range(self.reward_model_wg_size))

                    overhead_started_s = time.monotonic()
                    engine_wakeup_s = 0.0
                    router_resume_s = 0.0
                    success = False
                    try:
                        stage_started_s = time.monotonic()
                        await asyncio.gather(
                            *[
                                self.reward_model_wg_list[instance_id].execute_rank_zero_async("wake_up")
                                for instance_id in instance_ids
                            ]
                        )
                        engine_wakeup_s = time.monotonic() - stage_started_s
                        if self.reward_model_router is not None:
                            stage_started_s = time.monotonic()
                            await self.reward_model_router.resume_instances.remote(instance_ids)
                            router_resume_s = time.monotonic() - stage_started_s
                    except Exception:
                        psrl_logger.exception("WAKE_UP command failed for RM instances %s", instance_ids)
                        self._complete_command(command_id, False)
                    else:
                        success = True
                        psrl_logger.info("WAKE_UP command for instances %s completed.", instance_ids)
                        self._complete_command(command_id, True)
                    finally:
                        psrl_logger.info(
                            "[ELASTIC_OVERHEAD] operation=wakeup role=RewardModel instances=%s success=%s "
                            "engine_wakeup_s=%.6f router_resume_s=%.6f total_s=%.6f",
                            instance_ids,
                            success,
                            engine_wakeup_s,
                            router_resume_s,
                            time.monotonic() - overhead_started_s,
                        )
                else:
                    raise ValueError(f"Unknown command type: {command_type}")

                self._active_command_id = None
            await asyncio.sleep(0)

        psrl_logger.info("Background command handler of reward model coordinator has finished.")

    def _on_command_handler_done(self, task: asyncio.Task) -> None:
        """Fail the active command and restart the queue loop after an unexpected exception."""
        if task.cancelled():
            return
        exception = task.exception()
        if exception is None:
            return
        psrl_logger.error(
            "Reward command handler crashed; active_command_id=%s. Restarting queue loop.",
            self._active_command_id,
            exc_info=(type(exception), exception, exception.__traceback__),
        )
        if self._active_command_id is not None:
            self._complete_command(self._active_command_id, False)
            self._active_command_id = None
        if not self.stop_command_handler and self.running_loop is not None:
            self.command_handler_task = self.running_loop.create_task(self._command_handler_loop())
            self.command_handler_task.add_done_callback(self._on_command_handler_done)

    async def _process_status_queue(self, instance_id: int):
        psrl_logger.info(f"Starting to process status queue for instance {instance_id}")
        while not self.stop_process_status_queue[instance_id]:
            # TODO(lhy) (from rollout coordinator):
            # add timeout handling for future fault tolerance of rollout instances
            recv_stats = await self.status_queues[instance_id].get_async(block=True, timeout=None)
            self.instance_to_engine_status[instance_id] = recv_stats
            psrl_logger.debug(
                f"Updated engine status for instance "
                f"{recv_stats.instance_id}: {self.instance_to_engine_status[instance_id]}"
            )

    async def _broadcast_status_to_router(self):
        """Broadcast collected engine status to the reward-model router.

        Mirrors ``RolloutCoordinator._broadcast_status_to_router``. Unlike the
        rollout coordinator, the RM router handle is set after ``start_busy_loop``
        returns, so this loop waits for the router to appear before pushing.
        """
        while not self.stop_broadcast_status_to_router:
            if self.reward_model_router is None:
                await asyncio.sleep(0.5)
                continue
            await asyncio.sleep(self.config.psrl.status_collection.coordinator_sync_interval_in_ms / 1000)
            try:
                await self.reward_model_router.update_instance_status.remote(self.instance_to_engine_status)
            except Exception as exc:
                psrl_logger.warning(
                    "Failed to broadcast RM engine status to router: %s", exc
                )

    def get_instance_engine_status_snapshot(self) -> dict[int, dict]:
        """
        Return a lightweight snapshot map for elastic scaling decisions.
        """
        result: dict[int, dict] = {}
        for instance_id, engine_stats in self.instance_to_engine_status.items():
            result[int(instance_id)] = {
                "instance_id": int(engine_stats.instance_id),
                "model_version": int(engine_stats.model_version),
                "timestamp": engine_stats.snapshot.get("timestamp"),
                "scheduler_stats": engine_stats.snapshot.get("scheduler_stats", {}),
                "generation_throughput": float(engine_stats.snapshot.get("generation_throughput", 0.0)),
            }
        return result

    async def get_router_backlog_size(self) -> int:
        """Return pending request count in reward-model router queue."""
        t_enter = time.monotonic()
        model_tag = self.reward_model_name
        log_elastic_rm_backlog_diag(
            psrl_logger,
            "stage=RewardModelCoordinator_enter model=%s elapsed_since_entry_s=0.000",
            model_tag,
        )
        if self.reward_model_router is None:
            return 0
        log_elastic_rm_backlog_diag(
            psrl_logger,
            "stage=RewardModelCoordinator_before_router_rpc model=%s since_enter_s=%.3f",
            model_tag,
            time.monotonic() - t_enter,
        )
        t_rpc = time.monotonic()
        pending = int(await self.reward_model_router.get_pending_request_count.remote())
        log_elastic_rm_backlog_diag(
            psrl_logger,
            "stage=RewardModelCoordinator_after_router_rpc model=%s pending=%d router_rpc_s=%.3f since_enter_s=%.3f",
            model_tag,
            pending,
            time.monotonic() - t_rpc,
            time.monotonic() - t_enter,
        )
        return pending

    async def get_router_backlog_summary(self, top_t: int | None = None) -> dict[str, int]:
        """Return selected reward router waiting queue load for elastic scaling."""
        t_enter = time.monotonic()
        model_tag = self.reward_model_name
        log_elastic_rm_backlog_diag(
            psrl_logger,
            "stage=RewardModelCoordinator_summary_enter model=%s elapsed_since_entry_s=0.000",
            model_tag,
        )
        if self.reward_model_router is None:
            return {"pending": 0, "count": 0, "total_tokens": 0}
        log_elastic_rm_backlog_diag(
            psrl_logger,
            "stage=RewardModelCoordinator_summary_before_router_rpc model=%s since_enter_s=%.3f",
            model_tag,
            time.monotonic() - t_enter,
        )
        t_rpc = time.monotonic()
        summary = await self.reward_model_router.get_pending_request_summary.remote(top_t)
        log_elastic_rm_backlog_diag(
            psrl_logger,
            (
                "stage=RewardModelCoordinator_summary_after_router_rpc model=%s pending=%d count=%d "
                "total_tokens=%d router_rpc_s=%.3f since_enter_s=%.3f"
            ),
            model_tag,
            int(summary.get("pending", 0)),
            int(summary.get("count", 0)),
            int(summary.get("total_tokens", 0)),
            time.monotonic() - t_rpc,
            time.monotonic() - t_enter,
        )
        return summary

    async def get_candidate_evaluation_snapshot(self, top_t: int | None = None) -> dict:
        """Proxy the compact request-level snapshot from the RM router."""
        if self.reward_model_router is None:
            raise RuntimeError("reward-model router is not initialized")
        return await self.reward_model_router.get_candidate_evaluation_snapshot.remote(top_t)

    async def init_route_strategy(self):
        # TODO(zyf): need to decide whether to use the route strategy for rm
        pass

        # await self._is_init_model.wait()
        # futures = []
        # for i in range(self.config.psrl.deployment.n_rollout_instances):
        #     if self.rank_0_is_model_owner:
        #         futures.append(self.rollout_wg_list[i].execute_rank_zero_async("estimate_max_model_len"))
        #     else:
        #         futures.extend(self.rollout_wg_list[i].execute_all_async("estimate_max_model_len"))
        # max_model_lens = await asyncio.gather(*futures)
        # psrl_logger.info(f"Max model lens: {max_model_lens}")
        # instance_to_max_model_len = {
        #     i: max(max_model_lens[i]) for i in range(self.config.psrl.deployment.n_rollout_instances)
        # }
        # # Use the max model len to budget the kv cache size for each instance
        # futures = []
        # for agent_worker in self.agent_loop_workers:
        #     futures.append(
        #         agent_worker.init_route_strategy.remote(
        #             instance_to_max_model_len=instance_to_max_model_len,
        #         )
        #     )
        # await asyncio.gather(*futures)
