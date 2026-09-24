"""
NIXL name string constants for PivotRL.

This module is the single source of truth for all NIXL string identifiers
used to register agents and clients with the NIXL meta server. Both the NIXL
infrastructure layer (pivotrl.utils.nixl) and the worker naming layer
(pivotrl.utils.common.worker_naming) import from here.
"""

# Meta server name — the NIXL coordination server all agents register with.
NIXL_META_SERVER_NAME = "NIXLMetaServer"

# Client name prefixes — used by worker_naming to construct per-worker client names.
NIXL_GEN_CLIENT_PREFIX = "NIXLGenClient"
NIXL_TRAIN_CLIENT_PREFIX = "NIXLTrainClient"
NIXL_PS_CLIENT_PREFIX = "NIXLPSClient"
