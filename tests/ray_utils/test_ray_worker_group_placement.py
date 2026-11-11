"""
Verify worker ranks and local device counts on unequal placement groups.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from verl.single_controller.ray import base

pytestmark = pytest.mark.cpu_test


def _worker_group():
    group = base.RayWorkerGroup.__new__(base.RayWorkerGroup)
    group.device_name = "cuda"
    group.use_gpu = True
    group.name_prefix = "heterogeneous"
    group._ray_master_port_range = None
    group._get_master_addr_port = Mock()
    return group


def test_full_group_creates_all_ranks_after_placement_group_sorting(monkeypatch):
    eight = SimpleNamespace(id="eight", bundle_count=8)
    four = SimpleNamespace(id="four", bundle_count=4)
    pool = SimpleNamespace(world_size=12, store=[8, 4], get_placement_groups=lambda **kwargs: [eight, four])
    group = _worker_group()
    group._create_worker = Mock()
    monkeypatch.setattr(base, "sort_placement_group_by_node_ip", lambda groups: [four, eight])

    group._init_with_resource_pool(pool, object(), bin_pack=True, detached=False)

    calls = [call.kwargs for call in group._create_worker.call_args_list]
    assert len(calls) == 12, "Every allocated GPU must receive exactly one worker."
    assert [call["rank"] for call in calls] == list(range(12)), "Global ranks must remain contiguous."
    assert [(call["pg"].id, call["local_rank"]) for call in calls] == [
        *[("four", rank) for rank in range(4)],
        *[("eight", rank) for rank in range(8)],
    ], "Local ranks must follow each placement group's actual capacity."
    group._get_master_addr_port.assert_called_once_with(four, bundle_index=0, master_port_range=None)


def test_subgroup_master_uses_variable_placement_group_offsets():
    four = SimpleNamespace(id="four", bundle_count=4)
    eight = SimpleNamespace(id="eight", bundle_count=8)
    pool = SimpleNamespace(
        world_size=2,
        store=[8, 4],
        start_bundle_index=5,
        get_placement_groups=lambda **kwargs: [four, eight],
    )
    group = _worker_group()
    group._create_worker = Mock()

    group._init_with_subresource_pool(pool, object(), bin_pack=True, detached=False)

    group._get_master_addr_port.assert_called_once_with(eight, bundle_index=1, master_port_range=None)
    assert [call.kwargs["local_rank"] for call in group._create_worker.call_args_list] == [1, 2], (
        "A subgroup must resolve its bundle slice across variable placement group sizes."
    )


@pytest.mark.parametrize("local_world_size", [4, 8])
def test_worker_environment_uses_its_own_placement_group_size(local_world_size):
    group = _worker_group()
    group._master_addr = "192.0.2.1"
    group._master_port = "12345"
    group.profile_steps = None
    group._workers = []
    group._worker_names = []
    pool = SimpleNamespace(world_size=12, store=[8, 4], use_gpu=True, resource_num_per_bundle=1, max_colocate_count=1)
    factory = Mock()
    factory.cls = object

    group._create_worker(
        rank=9,
        pg_idx=1,
        pg=SimpleNamespace(bundle_count=local_world_size),
        local_rank=1,
        resource_pool=pool,
        ray_cls_with_init=factory,
        worker_env=None,
        detached=False,
    )

    options = factory.update_options.call_args.args[0]
    environment = options["runtime_env"]["env_vars"]
    assert environment["WORLD_SIZE"] == "12", "Each worker must retain the global training world size."
    assert environment["RAY_LOCAL_WORLD_SIZE"] == str(local_world_size), (
        "Worker environment metadata must describe its actual placement group."
    )
    assert environment["RAY_LOCAL_RANK"] == "1", "Local rank must retain its bundle index."
