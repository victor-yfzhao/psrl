import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch
from psrl.utils.transfer_queue import payload
from psrl.utils.transfer_queue.payload import PayloadState
from tensordict import TensorDict
from transfer_queue import BatchMeta, KVBatchMeta
from transfer_queue.metadata import extract_field_schema
from verl.protocol import BatchData
from verl.single_controller.base.decorator import make_nd_compute_dataproto_dispatch_fn
from verl.single_controller.base.worker import Worker
from verl.utils import transferqueue_utils

pytestmark = pytest.mark.cpu_test


def test_clear_payload_filters_missing_keys(monkeypatch):
    kv_list = MagicMock(return_value={"train": {"34_0": {}, "34_1": {}}})
    kv_clear = MagicMock()
    monkeypatch.setattr(payload.tq, "kv_list", kv_list)
    monkeypatch.setattr(payload.tq, "kv_clear", kv_clear)

    payload.clear_payload(
        keys=["34", "34_0", "34_1"],
        partition_id="train",
        state=PayloadState.DROPPED,
    )

    kv_clear.assert_called_once_with(keys=["34_0", "34_1"], partition_id="train")


def test_async_clear_payload_skips_absent_keys(monkeypatch):
    async_kv_list = AsyncMock(return_value={"train": {"other": {}}})
    async_kv_clear = AsyncMock()
    monkeypatch.setattr(payload.tq, "async_kv_list", async_kv_list)
    monkeypatch.setattr(payload.tq, "async_kv_clear", async_kv_clear)

    asyncio.run(
        payload.async_clear_payload(
            keys=["34"],
            partition_id="train",
            state=PayloadState.DROPPED,
        )
    )

    async_kv_clear.assert_not_awaited()


@pytest.mark.parametrize("need_collect", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("kv_input", [False, True])
def test_bridge_returns_only_collected_results(monkeypatch, need_collect, asynchronous, kv_input):
    class CollectWorker(Worker):
        def __init__(self):
            self.collect = need_collect

        def query_collect_info(self, mesh_name):
            assert mesh_name == "actor", "The bridge must use its registered mesh."
            return self.collect

    def compute(worker, data):
        return TensorDict({"log_probs": data["tokens"].float()}, batch_size=data.batch_size)

    async def async_compute(worker, data):
        return compute(worker, data)

    async def get_data(meta):
        return TensorDict({"tokens": torch.tensor(meta.global_indexes)}, batch_size=[meta.size])

    async def put_data(data, metadata):
        return BatchMeta(
            global_indexes=metadata.global_indexes,
            partition_ids=metadata.partition_ids,
            field_schema=extract_field_schema(data),
        )

    async def retrieve_meta(keys, partition_id, create):
        assert not create, "Input metadata must already exist."
        return BatchMeta(
            global_indexes=[int(key.removeprefix("row-")) for key in keys],
            partition_ids=[partition_id] * len(keys),
        )

    async def retrieve_keys(global_indexes, partition_id):
        assert partition_id == "training", "Result metadata must retain its partition."
        return [f"row-{index}" for index in global_indexes]

    client = SimpleNamespace(
        async_get_data=AsyncMock(side_effect=get_data),
        async_put=AsyncMock(side_effect=put_data),
        async_kv_retrieve_meta=AsyncMock(side_effect=retrieve_meta),
        async_kv_retrieve_keys=AsyncMock(side_effect=retrieve_keys),
    )
    monkeypatch.setattr(transferqueue_utils, "TQ_INITIALIZED", True)
    monkeypatch.setattr(transferqueue_utils.tq, "get_client", lambda: client)
    indexes = list(range(4))
    meta = (
        KVBatchMeta(keys=[f"row-{index}" for index in indexes], tags=[{}] * 4, partition_id="training")
        if kv_input
        else BatchMeta(global_indexes=indexes, partition_ids=["training"] * 4)
    )
    function = transferqueue_utils.tqbridge(make_nd_compute_dataproto_dispatch_fn("actor"))(
        async_compute if asynchronous else compute
    )
    result = function(CollectWorker(), meta)
    if asynchronous:
        result = asyncio.run(result)

    if need_collect:
        assert result.size == 4, "The collector must return every input row."
        fields = result.fields if kv_input else result.field_names
        assert fields == ["log_probs"], "Result metadata must describe the computed field."
        if kv_input:
            assert result.keys == meta.keys, "Collected keys must preserve input order."
        else:
            assert result.global_indexes == indexes, "Collected indexes must preserve input order."
            collected = BatchData([result]).concat()
            assert collected.keys == [f"row-{index}" for index in indexes], (
                "Controller collection must preserve unique ordered keys."
            )
            assert collected.fields == ["log_probs"], "Controller collection must preserve result fields."
        client.async_put.assert_awaited_once()
    else:
        assert isinstance(result, BatchMeta) and result.size == 0, "Other ranks must return valid empty metadata."
        client.async_put.assert_not_awaited()
