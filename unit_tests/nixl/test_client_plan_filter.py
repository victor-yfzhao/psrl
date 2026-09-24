import sys
import types
from unittest.mock import Mock

import pytest

if "ray" not in sys.modules:
    ray_stub = types.ModuleType("ray")
    ray_actor_stub = types.ModuleType("ray.actor")

    class _ActorHandle:
        pass

    class _ObjectRef:
        pass

    ray_actor_stub.ActorHandle = _ActorHandle
    ray_stub.actor = ray_actor_stub
    ray_stub.ObjectRef = _ObjectRef

    def _remote(cls):
        class _RemoteActor:
            @staticmethod
            def remote(*args, **kwargs):
                return cls(*args, **kwargs)

        return _RemoteActor

    ray_stub.remote = _remote
    sys.modules["ray"] = ray_stub
    sys.modules["ray.actor"] = ray_actor_stub

from pivotrl.utils.nixl.client import NIXLStorageClient
from pivotrl.utils.nixl.comm_plan import NIXLCommPlan
from pivotrl.utils.nixl.nixl_spec import NIXLClientType


@pytest.mark.parametrize(
    ("client_type", "operation", "plan"),
    [
        (
            NIXLClientType.PULL_SIDE,
            "client_read",
            NIXLCommPlan({}, {"client": {"weight": {"assigned_ps": [(0,)]}}}, {}),
        ),
        (
            NIXLClientType.PUSH_SIDE,
            "client_read",
            NIXLCommPlan({}, {}, {"client": {"weight": {"assigned_ps": [(0,)]}}}),
        ),
        (
            NIXLClientType.PUSH_SIDE,
            "client_write",
            NIXLCommPlan({"client": {"weight": {"assigned_ps": [(0,)]}}}, {}, {}),
        ),
    ],
)
def test_unplanned_target_is_skipped_before_client_info_lookup(client_type, operation, plan):
    client = NIXLStorageClient.__new__(NIXLStorageClient)
    client.client_name = "client"
    client.client_type = client_type
    client._comm_plan = plan
    client._ensure_client_info_fetched = Mock(side_effect=AssertionError("unexpected client-info lookup"))

    result = getattr(client, operation)("unused_agent", "unused_ps", "weight", "test")

    assert result == []
    client._ensure_client_info_fetched.assert_not_called()
