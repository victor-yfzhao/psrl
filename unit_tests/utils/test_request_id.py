from pivotrl.utils.rollout.request_id import canonical_pivotrl_request_id


def test_canonical_pivotrl_request_id_strips_only_vllm_internal_suffix():
    assert canonical_pivotrl_request_id("389-b51b174a") == "389"
    assert canonical_pivotrl_request_id(389) == "389"
    assert canonical_pivotrl_request_id("request-with-hyphens") == "request-with-hyphens"
