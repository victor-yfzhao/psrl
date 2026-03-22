import asyncio
import enum
import logging
import os
from enum import Enum

import ray

from psrl.trainer.ppo.utils import PSRL_Role
from psrl.utils.server.command import Command, CommandType

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class InstanceStatus(Enum):
    ASLEEP = enum.auto()
    AWAKEN = enum.auto()


@ray.remote
class ElasticExecutor:
    def __init__(
        self,
        roles: list[tuple[PSRL_Role, str]],
        coordinators: dict[PSRL_Role, dict[str, ray.actor.ActorHandle]],
    ):
        self.coordinators = coordinators
        self.roles = roles

        self.instances_status_flags: dict[PSRL_Role, dict[str, dict[int, InstanceStatus]]] = {}
        self.instances_engine_stats: dict[PSRL_Role, dict[str, dict[int, dict | None]]] = {}
        self.instance_gpu_mappings: dict[PSRL_Role, dict[str, dict[int, dict[str, object]]]] = {}
        self.gpu_to_instances: dict[tuple[str | None, int], set[tuple[PSRL_Role, str, int]]] = {}

        for role_name, model_name in self.roles:
            self.instances_status_flags.setdefault(role_name, {}).setdefault(model_name, {})
            self.instances_engine_stats.setdefault(role_name, {}).setdefault(model_name, {})
            self.instance_gpu_mappings.setdefault(role_name, {}).setdefault(model_name, {})

        self.scale_up_task_queue: asyncio.Queue = asyncio.Queue()
        self.scale_down_task_queue: asyncio.Queue = asyncio.Queue()

        self.running_loop = None
        self.monitor_task = None
        self.scale_up_task = None
        self.scale_down_task = None

        self.stop_monitor = False
        self.stop_scale_up = False
        self.stop_scale_down = False

    def register_instances(self, role_name: PSRL_Role, model_name: str, num_instances: int):
        self.instances_status_flags.setdefault(role_name, {}).setdefault(model_name, {})
        self.instances_engine_stats.setdefault(role_name, {}).setdefault(model_name, {})
        self.instance_gpu_mappings.setdefault(role_name, {}).setdefault(model_name, {})
        for instance_id in range(num_instances):
            self.instances_status_flags[role_name][model_name][instance_id] = InstanceStatus.ASLEEP
            self.instances_engine_stats[role_name][model_name][instance_id] = None
            self.instance_gpu_mappings[role_name][model_name].setdefault(
                instance_id,
                {"gpu_ids": [], "node_id": None},
            )

    def register_instance_gpu_mapping(
        self,
        role_name: PSRL_Role,
        model_name: str,
        instance_id: int,
        gpu_ids: list[int] | None,
        node_id: str | None = None,
    ):
        self.instance_gpu_mappings.setdefault(role_name, {}).setdefault(model_name, {})
        # Clear historical reverse index before updating mapping.
        self._remove_instance_from_gpu_reverse_index(role_name, model_name, instance_id)
        self.instance_gpu_mappings[role_name][model_name][instance_id] = {
            "gpu_ids": list(gpu_ids or []),
            "node_id": node_id,
        }
        self._add_instance_to_gpu_reverse_index(role_name, model_name, instance_id)

    def initialize_instance_states(
        self,
        awaken_instances: list[dict] | None = None,
    ):
        awaken_set = set()
        for item in awaken_instances or []:
            awaken_set.add((item["role_name"], item["model_name"], int(item["instance_id"])))

        for role_name, role_data in self.instances_status_flags.items():
            for model_name, instance_status in role_data.items():
                for instance_id in list(instance_status.keys()):
                    key = (role_name, model_name, int(instance_id))
                    if key in awaken_set:
                        self.instances_status_flags[role_name][model_name][instance_id] = InstanceStatus.AWAKEN
                    else:
                        self.instances_status_flags[role_name][model_name][instance_id] = InstanceStatus.ASLEEP

    def initialize_runtime(
        self,
        registrations: list[dict] | None = None,
        gpu_mappings: list[dict] | None = None,
        awaken_instances: list[dict] | None = None,
    ):
        for item in registrations or []:
            self.register_instances(
                role_name=item["role_name"],
                model_name=item["model_name"],
                num_instances=int(item["num_instances"]),
            )
        for item in gpu_mappings or []:
            self.register_instance_gpu_mapping(
                role_name=item["role_name"],
                model_name=item["model_name"],
                instance_id=int(item["instance_id"]),
                gpu_ids=item.get("gpu_ids", []),
                node_id=item.get("node_id"),
            )
        self.initialize_instance_states(awaken_instances=awaken_instances)

    def snapshot(self) -> dict:
        return {
            "status": self.instances_status_flags,
            "gpu_mappings": self.instance_gpu_mappings,
            "gpu_to_instances": self.gpu_to_instances,
        }

    @staticmethod
    def _instance_key(role_name: PSRL_Role, model_name: str, instance_id: int) -> tuple[PSRL_Role, str, int]:
        return (role_name, model_name, int(instance_id))

    def _get_instance_gpu_keys(self, role_name: PSRL_Role, model_name: str, instance_id: int) -> list[tuple[str | None, int]]:
        gpu_mapping = self.instance_gpu_mappings.get(role_name, {}).get(model_name, {}).get(instance_id, {})
        node_id = gpu_mapping.get("node_id")
        gpu_ids = [int(gpu_id) for gpu_id in gpu_mapping.get("gpu_ids", [])]
        return [(node_id, gpu_id) for gpu_id in gpu_ids]

    def _remove_instance_from_gpu_reverse_index(self, role_name: PSRL_Role, model_name: str, instance_id: int):
        target_key = self._instance_key(role_name, model_name, instance_id)
        gpu_keys = self._get_instance_gpu_keys(role_name, model_name, instance_id)
        for gpu_key in gpu_keys:
            instance_set = self.gpu_to_instances.get(gpu_key)
            if not instance_set:
                continue
            instance_set.discard(target_key)
            if not instance_set:
                self.gpu_to_instances.pop(gpu_key, None)

    def _add_instance_to_gpu_reverse_index(self, role_name: PSRL_Role, model_name: str, instance_id: int):
        target_key = self._instance_key(role_name, model_name, instance_id)
        gpu_keys = self._get_instance_gpu_keys(role_name, model_name, instance_id)
        for gpu_key in gpu_keys:
            self.gpu_to_instances.setdefault(gpu_key, set()).add(target_key)

    def _has_other_role_awaken_on_shared_gpu(self, role_name: PSRL_Role, model_name: str, instance_id: int) -> bool:
        gpu_keys = self._get_instance_gpu_keys(role_name, model_name, instance_id)
        for gpu_key in gpu_keys:
            for other_role, other_model, other_instance_id in self.gpu_to_instances.get(gpu_key, set()):
                if other_role == role_name:
                    continue
                other_status = self.instances_status_flags.get(other_role, {}).get(other_model, {}).get(other_instance_id)
                if other_status == InstanceStatus.AWAKEN:
                    return True
        return False

    async def start_busy_loop(self):
        if self.monitor_task is not None and not self.monitor_task.done():
            return

        self.running_loop = asyncio.get_running_loop()
        self.stop_monitor = False
        self.stop_scale_up = False
        self.stop_scale_down = False

        self.monitor_task = self.running_loop.create_task(self._monitor_loop())
        self.monitor_task.add_done_callback(lambda f: f.result())

        self.scale_up_task = self.running_loop.create_task(self._scale_up_handler_loop())
        self.scale_up_task.add_done_callback(lambda f: f.result())

        self.scale_down_task = self.running_loop.create_task(self._scale_down_handler_loop())
        self.scale_down_task.add_done_callback(lambda f: f.result())

    async def stop(self):
        if self.monitor_task is None and self.scale_up_task is None and self.scale_down_task is None:
            return

        self.stop_monitor = True
        self.stop_scale_up = True
        self.stop_scale_down = True

        tasks_to_wait = []
        if self.monitor_task is not None:
            tasks_to_wait.append(self.monitor_task)
        if self.scale_up_task is not None:
            tasks_to_wait.append(self.scale_up_task)
        if self.scale_down_task is not None:
            tasks_to_wait.append(self.scale_down_task)
        if tasks_to_wait:
            await asyncio.gather(*tasks_to_wait, return_exceptions=True)

    async def _monitor_loop(self):
        while not self.stop_monitor:
            await asyncio.sleep(0)

    async def _scale_up_handler_loop(self):
        while not self.stop_scale_up:
            if self.scale_up_task_queue.empty():
                await asyncio.sleep(0)
                continue

            role_need_to_scale_up = self.scale_up_task_queue.get_nowait()
            instances_to_scaled_down = self._find_instances_to_scaled_down_for_other_roles(role_need_to_scale_up)
            if instances_to_scaled_down:
                await asyncio.gather(
                    *[self._scale_down_instance(instance) for instance in instances_to_scaled_down]
                )

            instances_to_scaled_up = self._find_instances_to_scaled_up(role_need_to_scale_up)
            if not instances_to_scaled_up:
                psrl_logger.warning("No instances can be scaled up for role %s", role_need_to_scale_up)
                continue
            await asyncio.gather(*[self._scale_up_instance(instance) for instance in instances_to_scaled_up])

    async def _scale_down_handler_loop(self):
        while not self.stop_scale_down:
            if self.scale_down_task_queue.empty():
                await asyncio.sleep(0)
                continue

            role_need_to_scale_down = self.scale_down_task_queue.get_nowait()
            instances_to_scaled_down = self._find_instances_to_scaled_down_in_role(role_need_to_scale_down)
            if not instances_to_scaled_down:
                psrl_logger.warning("No instances can be scaled down for role %s", role_need_to_scale_down)
                continue
            await asyncio.gather(*[self._scale_down_instance(instance) for instance in instances_to_scaled_down])

    async def _scale_up_instance(self, instance_to_scaled_up: dict):
        instance_role = instance_to_scaled_up["role_name"]
        instance_model_name = instance_to_scaled_up["model_name"]
        instance_id = int(instance_to_scaled_up["instance_id"])

        coordinator = self.coordinators[instance_role][instance_model_name]
        await coordinator.exec_command.remote(
            Command(
                type=CommandType.WAKE_UP,
                instance_ids=[instance_id],
            ),
            blocking=True,
        )
        self.instances_status_flags[instance_role][instance_model_name][instance_id] = InstanceStatus.AWAKEN

    async def _scale_down_instance(self, instance_to_scaled_down: dict):
        instance_role = instance_to_scaled_down["role_name"]
        instance_model_name = instance_to_scaled_down["model_name"]
        instance_id = int(instance_to_scaled_down["instance_id"])

        coordinator = self.coordinators[instance_role][instance_model_name]
        await coordinator.exec_command.remote(
            Command(
                type=CommandType.SLEEP,
                instance_ids=[instance_id],
            ),
            blocking=True,
        )
        self.instances_status_flags[instance_role][instance_model_name][instance_id] = InstanceStatus.ASLEEP
        self.instances_engine_stats[instance_role][instance_model_name][instance_id] = None

    def _find_instances_to_scaled_up(self, role_need_to_scale_up: dict):
        role_name = role_need_to_scale_up["role_name"]
        model_name = role_need_to_scale_up["model_name"]
        num_instances = int(role_need_to_scale_up.get("num_instances", 1))
        status_dict = self.instances_status_flags.get(role_name, {}).get(model_name, {})
        candidate_ids = [instance_id for instance_id, status in status_dict.items() if status == InstanceStatus.ASLEEP]
        if not candidate_ids:
            return None
        filtered_ids = [
            instance_id
            for instance_id in candidate_ids
            if not self._has_other_role_awaken_on_shared_gpu(role_name, model_name, int(instance_id))
        ]
        if not filtered_ids:
            return None
        return [
            {"role_name": role_name, "model_name": model_name, "instance_id": instance_id}
            for instance_id in filtered_ids[:num_instances]
        ]

    def _find_instances_to_scaled_down_for_other_roles(self, role_need_to_scale_up: dict):
        target_role = role_need_to_scale_up["role_name"]
        num_instances = int(role_need_to_scale_up.get("num_instances", 1))
        candidates: list[dict] = []

        for role_name, model_name in self.roles:
            if role_name == target_role:
                continue
            role_status = self.instances_status_flags.get(role_name, {}).get(model_name, {})
            for instance_id, status in role_status.items():
                if status == InstanceStatus.AWAKEN:
                    candidates.append(
                        {"role_name": role_name, "model_name": model_name, "instance_id": int(instance_id)}
                    )

        if not candidates:
            return None
        return candidates[:num_instances]

    def _find_instances_to_scaled_down_in_role(self, role_need_to_scale_down: dict):
        role_name = role_need_to_scale_down["role_name"]
        model_name = role_need_to_scale_down["model_name"]
        num_instances = int(role_need_to_scale_down.get("num_instances", 1))
        role_status = self.instances_status_flags.get(role_name, {}).get(model_name, {})
        candidate_ids = [instance_id for instance_id, status in role_status.items() if status == InstanceStatus.AWAKEN]
        if not candidate_ids:
            return None
        return [
            {"role_name": role_name, "model_name": model_name, "instance_id": int(instance_id)}
            for instance_id in candidate_ids[:num_instances]
        ]
