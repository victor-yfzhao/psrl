# PSRL Docker images

The GPU and CPU images clone PSRL from GitHub. `PSRL_COMMIT` selects a branch,
tag, or commit and defaults to `main`. Build from the repository root.
No extra `pip install` step is required after building.

## Prerequisites

- Docker with BuildKit enabled.
- For the GPU image: an NVIDIA driver (R580 or newer) and the NVIDIA Container
  Toolkit.
- Sufficient disk space and build time for the native CUDA dependencies.

## Build an image

Run these commands from the repository root.

### GPU image

```bash
docker build --progress=plain \
  -f docker/Dockerfile \
  -t psrl:latest .
```

The GPU image includes CUDA 13.0.3, PyTorch 2.11, vLLM, veRL, NIXL/UCX,
LMCache, Megatron, SMG, and the PSRL runtime. It is the image to use for
training and multi-node rollout.

To build the current commit, push it to the PSRL GitHub repository first:

```bash
docker build --progress=plain \
  --build-arg PSRL_COMMIT="$(git rev-parse HEAD)" \
  -f docker/Dockerfile \
  -t psrl:latest .
```

The build context contains only Docker build inputs and dependency patches,
as listed in `.dockerignore`. PSRL source comes from the selected Git revision.
Local uncommitted changes are not included. Mount datasets and checkpoints
when starting a container.

### CPU image

```bash
docker buildx build --load --progress=plain \
  --platform=linux/amd64 \
  -f docker/Dockerfile.cpu \
  -t psrl:cpu .
```

The CPU image is intended for imports, configuration checks, unit tests, and
gateway/data-plane development. It does not support PSRL GPU training.

## Start a container

The shell starts in the installed PSRL checkout.

```bash
docker run --rm --gpus all --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -it psrl:latest
```

Inside the container, activate the runtime environment and check the imports:

```bash
source env/psrl.sh
python -c 'import psrl, torch, nixl; print(torch.__version__)'
```

For a CPU shell, omit `--gpus all` and use the CPU tag:

```bash
docker run --rm -it \
  --ipc=host \
  --network=host \
  psrl:cpu
```

For multi-node NIXL/UCX/Mooncake traffic, start the same image on every node and
expose the host's RDMA devices through Docker. Supply `UCX_NET_DEVICES`,
`NCCL_SOCKET_IFNAME`, and `NCCL_IB_HCA` through the deployment environment when
interface selection is needed. The image does not select network devices.

## Notes

- The image starts a shell and does not launch Ray or a training job
  automatically.
- The GPU image uses a CUDA 13.0 / PyTorch `cu130` stack. If you override
  `CUDA_VERSION` or `BASE_IMAGE`, revalidate the matching PyTorch wheel and
  constraints in `docker/constraints.txt`.
- Both images build patched UCX 1.20 and NIXL 1.2. The CPU image builds NIXL
  without CUDA support. Its native distribution is named `nixl-cu12` upstream.
- The selected PSRL revision must include the NIXL 1.2 initialization support.
  Commit and push local changes before selecting them with `PSRL_COMMIT`.
- When changing a pinned third-party revision, rebase the matching patch in
  `patch/` and rebuild the image from a clean cache.
