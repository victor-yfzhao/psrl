#!/usr/bin/env python
"""
Comprehensive test for communication planner with custom sorting algorithm.

Requires: torch, ray, and the `nixl` Python extension (via `import psrl.utils.nixl`).
If Ray is unavailable, a minimal stub is installed below so `PortScanner.remote()` can import;
you still need `nixl` for `psrl.utils.nixl.client` to load.
"""

import os
import sys
import types
from collections import OrderedDict

# `global_vars` imports `PortScanner.remote()` at import time; allow running without Ray.
if "ray" not in sys.modules:
    _ray = types.ModuleType("ray")
    _ray_actor = types.ModuleType("ray.actor")

    class _ActorHandle:
        pass

    _ray_actor.ActorHandle = _ActorHandle
    sys.modules["ray.actor"] = _ray_actor
    _ray.actor = _ray_actor

    def _remote_decorator(_cls=None, **_kwargs):
        def _decorate(cls):
            class _Actor:
                @staticmethod
                def remote(*_a, **_kw):
                    return cls()

            _Actor.__name__ = cls.__name__
            return _Actor

        return _decorate(_cls) if _cls is not None else _decorate

    _ray.remote = _remote_decorator
    sys.modules["ray"] = _ray

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from psrl.utils.nixl.comm_plan import global_comm_planner
from psrl.utils.nixl.network_topology import NetworkTopology
from psrl.utils.nixl.nixl_spec import (
    NIXLClientInfo,
    NIXLClientType,
    NIXLShardMetaInfo,
    NIXLSharding,
    NIXLTensorInfo,
)


def _make_weight_tensor_info(global_shard_indices: list[int], total_global_shards: int = 4) -> NIXLTensorInfo:
    """Build a valid NIXLTensorInfo: 1D shard along dim 0, equal-sized shards."""
    shard_mesh = OrderedDict([(0, total_global_shards)])
    shard_indices = [(i,) for i in global_shard_indices]
    rows = 1000 // total_global_shards
    shape = torch.Size((rows, 1000))
    stride = (1000, 1)
    desc_bytes_list = [f"shard_{i}".encode() for i in global_shard_indices]
    temp_desc_bytes_list = [b""] * len(global_shard_indices)
    shard_meta_infos = [
        NIXLShardMetaInfo(
            dtype=torch.float32,
            device=torch.device("cuda", 0),
            shape=shape,
            stride=stride,
            is_contiguous=True,
        )
        for _ in global_shard_indices
    ]
    sharding = NIXLSharding(shard_mesh=shard_mesh, shard_indices=shard_indices)
    return NIXLTensorInfo(
        desc_bytes_list=desc_bytes_list,
        temp_desc_bytes_list=temp_desc_bytes_list,
        sharding=sharding,
        shard_meta_infos=shard_meta_infos,
    )


def create_test_tensor_desc_info_1():
    """This client holds global shards 0 and 1."""
    return _make_weight_tensor_info([0, 1])


def create_test_tensor_desc_info_2():
    """This client holds global shards 1 and 2."""
    return _make_weight_tensor_info([1, 2])


def create_test_tensor_desc_info_3():
    """This client holds global shards 0, 1 and 2."""
    return _make_weight_tensor_info([0, 1, 2])


def create_test_tensor_desc_info_ps():
    """PS holds global shards 0..2 (union of all PUSH_SIDE shards in this scenario)."""
    return _make_weight_tensor_info([0, 1, 2])


def test_communication_planner():
    """Test the communication planner with custom sorting"""
    print("Testing communication planner with custom sorting...")

    clients = {}

    clients["push_A"] = NIXLClientInfo(
        name="push_A",
        node_ip="192.168.1.1",
        node_gpu_id=0,
        type=NIXLClientType.PUSH_SIDE,
        tensor_infos={"weight": create_test_tensor_desc_info_1()},
        meta=b"push_meta_A",
    )

    clients["push_B"] = NIXLClientInfo(
        name="push_B",
        node_ip="192.168.1.1",
        node_gpu_id=1,
        type=NIXLClientType.PUSH_SIDE,
        tensor_infos={"weight": create_test_tensor_desc_info_2()},
        meta=b"push_meta_B",
    )

    clients["push_C"] = NIXLClientInfo(
        name="push_C",
        node_ip="192.168.1.1",
        node_gpu_id=-1,
        type=NIXLClientType.PUSH_SIDE,
        tensor_infos={"weight": create_test_tensor_desc_info_1()},
        meta=b"push_meta_C",
    )

    clients["push_D"] = NIXLClientInfo(
        name="push_D",
        node_ip="192.168.1.2",
        node_gpu_id=0,
        type=NIXLClientType.PUSH_SIDE,
        tensor_infos={"weight": create_test_tensor_desc_info_3()},
        meta=b"push_meta_D",
    )

    # Split PS: push path vs rollout pull path (matches production NIXL comm_plan).
    clients["ps_push"] = NIXLClientInfo(
        name="ps_push",
        node_ip="192.168.1.1",
        node_gpu_id=0,
        type=NIXLClientType.PS_FOR_PUSH,
        tensor_infos={"weight": create_test_tensor_desc_info_ps()},
        meta=b"ps_meta_push",
    )

    clients["ps_pull"] = NIXLClientInfo(
        name="ps_pull",
        node_ip="192.168.1.1",
        node_gpu_id=0,
        type=NIXLClientType.PS_FOR_PULL,
        tensor_infos={"weight": create_test_tensor_desc_info_ps()},
        meta=b"ps_meta_pull",
    )

    clients["pull_1"] = NIXLClientInfo(
        name="pull_1",
        node_ip="192.168.1.3",
        node_gpu_id=0,
        type=NIXLClientType.PULL_SIDE,
        tensor_infos={"weight": create_test_tensor_desc_info_1()},
        meta=b"pull_meta_1",
    )

    comm_plan = global_comm_planner.make_comm_plan(clients)

    print("\nGenerated communication plan:")
    print("=" * 50)

    print("PUSH_SIDE -> PS_FOR_PUSH Plan:")
    for push_client, key_plans in comm_plan.push_to_ps_plan.items():
        for key, ps_plans in key_plans.items():
            for ps_client, shards in ps_plans.items():
                link_type = global_comm_planner._get_link_type_for_test(push_client, ps_client)
                print(f"  {push_client} -> {ps_client}: {key} shards {shards} ({link_type.name})")

    print("\nPULL_SIDE <- PS_FOR_PULL Plan (rollout):")
    for pull_client, key_plans in comm_plan.rollout_pull_from_ps_plan.items():
        for key, ps_plans in key_plans.items():
            for ps_client, shards in ps_plans.items():
                link_type = global_comm_planner._get_link_type_for_test(ps_client, pull_client)
                print(f"  {ps_client} -> {pull_client}: {key} shards {shards} ({link_type.name})")

    print("\nPUSH_SIDE <- PS_FOR_PUSH Plan (train pull):")
    for push_client, key_plans in comm_plan.train_pull_from_ps_plan.items():
        for key, ps_plans in key_plans.items():
            for ps_client, shards in ps_plans.items():
                link_type = global_comm_planner._get_link_type_for_test(ps_client, push_client)
                print(f"  {ps_client} -> {push_client}: {key} shards {shards} ({link_type.name})")

    print("\nCommunication planner test completed!")


def test_network_topology_integration():
    """Test network topology integration with communication planner"""
    print("\nTesting network topology integration...")

    topology = NetworkTopology()

    topology.register_client("client_A", "192.168.1.1", 0)
    topology.register_client("client_B", "192.168.1.1", 1)
    topology.register_client("client_C", "192.168.1.1", -1)
    topology.register_client("client_D", "192.168.1.2", 0)

    print("Link type determination:")
    print(f"  client_A -> client_A: {topology.get_link_type('client_A', 'client_A').name}")
    print(f"  client_A -> client_B: {topology.get_link_type('client_A', 'client_B').name}")
    print(f"  client_A -> client_C: {topology.get_link_type('client_A', 'client_C').name}")
    print(f"  client_A -> client_D: {topology.get_link_type('client_A', 'client_D').name}")

    print("\nBandwidth values:")
    print(f"  LOCAL: {topology.get_bandwidth_gbps('client_A', 'client_A')} Gbps")
    print(f"  NVLINK: {topology.get_bandwidth_gbps('client_A', 'client_B')} Gbps")
    print(f"  PCIE: {topology.get_bandwidth_gbps('client_A', 'client_C')} Gbps")
    print(f"  IB: {topology.get_bandwidth_gbps('client_A', 'client_D')} Gbps")

    print("\nNetwork topology integration test completed!")


if __name__ == "__main__":
    test_network_topology_integration()
    test_communication_planner()
