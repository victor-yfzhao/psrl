from nixl._api import DEFAULT_COMM_PORT, nixl_agent, nixl_agent_config, nixl_thread_sync_t


def create_nixl_agent(
    agent_name: str,
    *,
    enable_prog_thread: bool = True,
    enable_listen_thread: bool = False,
    listen_port: int = DEFAULT_COMM_PORT,
    num_workers: int = 1,
) -> nixl_agent:
    """
    Create a UCX agent with PSRL's worker and synchronization policy.
    """
    if num_workers < 1:
        raise ValueError(f"num_workers must be at least 1, got {num_workers!r}.")

    config = nixl_agent_config(
        enable_prog_thread=enable_prog_thread,
        enable_listen_thread=enable_listen_thread,
        listen_port=listen_port,
        backends=[],
        sync_mode=nixl_thread_sync_t.NIXL_THREAD_SYNC_RW if num_workers > 1 else None,
    )
    agent = nixl_agent(agent_name, config)
    agent.create_backend(
        "UCX",
        {"num_workers": str(num_workers), "ucx_error_handling_mode": "none"},
    )
    return agent
