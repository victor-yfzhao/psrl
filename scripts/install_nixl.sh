#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UCX_VERSION="${UCX_VERSION:-v1.20.0}"
NIXL_VERSION="${NIXL_VERSION:-1.2.0}"
MAX_JOBS="${MAX_JOBS:-64}"
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
BUILD_ROOT="${NIXL_STACK_BUILD_ROOT:-${REPO_ROOT}/third_party/psrl-nixl-stack-1.2.0-ucx-1.20}"
UCX_SOURCE_DIR="${BUILD_ROOT}/ucx"
NIXL_SOURCE_DIR="${BUILD_ROOT}/nixl"
UCX_PREFIX="${UCX_PREFIX:-${REPO_ROOT}/third_party/ucx_1_20_rkey65535}"
NIXL_PREFIX="${NIXL_PREFIX:-${REPO_ROOT}/third_party/nixl_1_2_0_ucx_1_20_rkey65535_backlog4096}"
UCX_PATCH_FILE="${REPO_ROOT}/patch/ucx/v1.20.0.patch"
NIXL_PATCH_FILE="${REPO_ROOT}/patch/nixl/1.2.0.patch"
NIXL_WHEEL_DIR="${NIXL_WHEEL_DIR:-${BUILD_ROOT}/wheelhouse}"
NIXL_INSTALL_PYTHON="${NIXL_INSTALL_PYTHON:-true}"
NIXL_SUBPROJECT_CACHE_DIR="${NIXL_SUBPROJECT_CACHE_DIR:-}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-${BUILD_ROOT}/uv-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${BUILD_ROOT}/xdg-cache}"
export UV_NO_BUILD_ISOLATION="${UV_NO_BUILD_ISOLATION:-1}"

require_command() {
    local command_name="$1"

    if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "Required command is unavailable: ${command_name}." >&2
        exit 1
    fi
}

clone_source() {
    local repository="$1"
    local version="$2"
    local destination="$3"

    if [[ -e "${destination}" && ! -d "${destination}/.git" ]]; then
        echo "Source path exists but is not a Git repository: ${destination}." >&2
        exit 1
    fi
    if [[ ! -d "${destination}/.git" ]]; then
        git clone --depth 1 --branch "${version}" "${repository}" "${destination}"
    fi

    local expected_commit
    local actual_commit
    if ! expected_commit="$(git -C "${destination}" rev-parse "${version}^{commit}" 2>/dev/null)"; then
        echo "Existing source does not contain version ${version}: ${destination}." >&2
        exit 1
    fi
    actual_commit="$(git -C "${destination}" rev-parse HEAD)"
    if [[ "${actual_commit}" != "${expected_commit}" ]]; then
        echo "Existing source is not checked out at version ${version}: ${destination}." >&2
        exit 1
    fi
}

build_ucx() {
    if [[ -f "${UCX_PREFIX}/.psrl-ucx-1.20-worker-config-uint16-complete" ]]; then
        echo "Using the installed patched UCX ${UCX_VERSION}."
        return
    fi
    echo "Building UCX ${UCX_VERSION}."
    clone_source "https://github.com/openucx/ucx.git" "${UCX_VERSION}" "${UCX_SOURCE_DIR}"
    cd "${UCX_SOURCE_DIR}"

    if git apply --check "${UCX_PATCH_FILE}"; then
        git apply "${UCX_PATCH_FILE}"
    elif ! git apply --reverse --check "${UCX_PATCH_FILE}"; then
        echo "UCX source does not match the expected ${UCX_VERSION} patch context." >&2
        exit 1
    fi

    local types_file="src/ucp/core/ucp_types.h"
    grep -Eq '^typedef uint16_t[[:space:]]+ucp_worker_cfg_index_t;$' "${types_file}"
    grep -Eq '^#define[[:space:]]+UCP_WORKER_MAX_EP_CONFIG[[:space:]]+UINT16_MAX$' "${types_file}"
    grep -Eq '^#define[[:space:]]+UCP_WORKER_MAX_RKEY_CONFIG[[:space:]]+UINT16_MAX$' "${types_file}"
    grep -Eq '^#define[[:space:]]+UCP_WORKER_CFG_INDEX_NULL[[:space:]]+UINT16_MAX$' "${types_file}"
    grep -q 'ucs_array_length(&worker->rkey_config)' src/ucp/core/ucp_worker.c

    ./autogen.sh
    rm -rf build
    mkdir build
    cd build
    ../configure \
        --prefix="${UCX_PREFIX}" \
        --enable-shared \
        --disable-static \
        --disable-doxygen-doc \
        --enable-optimizations \
        --enable-cma \
        --enable-devel-headers \
        --without-go \
        --with-cuda="${CUDA_PATH}" \
        --with-verbs \
        --with-dm \
        --enable-mt \
        --with-rdmacm \
        --with-mlx5 \
        --with-ib-hw-tm
    make -j "${MAX_JOBS}"
    make install-strip

    touch "${UCX_PREFIX}/.psrl-ucx-1.20-worker-config-uint16-complete"
    LD_LIBRARY_PATH="${UCX_PREFIX}/lib:${UCX_PREFIX}/lib/ucx:${LD_LIBRARY_PATH:-}" \
        "${UCX_PREFIX}/bin/ucx_info" -v
}

install_nixl_meta_package() {
    local site so python_version
    local rpath="\$ORIGIN/../../../../lib64"

    for so in "${NIXL_PREFIX}"/lib/python*/site-packages/nixl_cu12/_bindings*.so; do
        [[ -f "${so}" ]] || continue
        patchelf --force-rpath --set-rpath "${rpath}" "${so}"
    done

    python_version="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    site="${NIXL_PREFIX}/lib/python${python_version}/site-packages"
    python -m pip install --no-deps --no-index --upgrade --target "${site}" \
        "${NIXL_SOURCE_DIR}/build/src/bindings/python/nixl-meta/nixl-${NIXL_VERSION}-py3-none-any.whl"
}

apply_nixl_patch() {
    cd "${NIXL_SOURCE_DIR}"
    if git apply --check "${NIXL_PATCH_FILE}" 2>/dev/null; then
        git apply "${NIXL_PATCH_FILE}"
    elif ! git apply --reverse --check "${NIXL_PATCH_FILE}"; then
        echo "NIXL source does not match the expected ${NIXL_VERSION} patch context." >&2
        exit 1
    fi

    grep -q 'listen(socketFd, 4096)' src/utils/stream/metadata_stream.cpp
    grep -q 'err_handling_mode = UCP_ERR_HANDLING_MODE_NONE' src/plugins/ucx/ucx_backend.cpp
    grep -q 'ucx_err_mode_to_string(UCP_ERR_HANDLING_MODE_NONE)' src/plugins/ucx/ucx_utils.cpp
    grep -q 'NIXL_THREAD_SYNC_RW' src/bindings/python/nixl_bindings.cpp
    grep -q 'sync_mode: Optional\[nixl_thread_sync_t\]' src/api/python/_api.py
}

build_nixl() {
    echo "Building NIXL ${NIXL_VERSION}."
    clone_source "https://github.com/ai-dynamo/nixl.git" "v${NIXL_VERSION}" "${NIXL_SOURCE_DIR}"
    apply_nixl_patch
    cd "${NIXL_SOURCE_DIR}"

    local wrap_args=()
    if [[ -n "${NIXL_SUBPROJECT_CACHE_DIR}" ]]; then
        if [[ ! -d "${NIXL_SUBPROJECT_CACHE_DIR}" ]]; then
            echo "NIXL subproject cache is unavailable: ${NIXL_SUBPROJECT_CACHE_DIR}." >&2
            exit 1
        fi
        mkdir -p subprojects/packagecache
        cp -a "${NIXL_SUBPROJECT_CACHE_DIR}/." subprojects/packagecache/
        wrap_args+=(--wrap-mode=nodownload)
    fi

    local setup_args=()
    if [[ -f build/meson-private/coredata.dat ]]; then
        setup_args+=(--reconfigure)
    fi
    meson setup build \
        "${setup_args[@]}" \
        --prefix="${NIXL_PREFIX}" \
        "${wrap_args[@]}" \
        -Dbuild_docs=false \
        -Dbuild_tests=false \
        -Dbuild_examples=false \
        -Denable_plugins=UCX,POSIX,GDS,GDS_MT \
        -Ducx_path="${UCX_PREFIX}" \
        -Dinstall_headers=true \
        -Ddisable_gds_backend=false
    meson compile -C build -j "${MAX_JOBS}"
    meson install -C build --no-rebuild
    install_nixl_meta_package

    mkdir -p "${NIXL_WHEEL_DIR}"
    python -m pip wheel \
        --no-deps \
        --no-build-isolation \
        --wheel-dir "${NIXL_WHEEL_DIR}" \
        --config-settings=build-dir="${BUILD_ROOT}/wheel-build" \
        --config-settings=compile-args="-j${MAX_JOBS}" \
        --config-settings=setup-args="-Ducx_path=${UCX_PREFIX}" \
        --config-settings=setup-args="-Dbuild_docs=false" \
        --config-settings=setup-args="-Dbuild_tests=false" \
        --config-settings=setup-args="-Dbuild_examples=false" \
        --config-settings=setup-args="-Denable_plugins=UCX,POSIX,GDS,GDS_MT" \
        --config-settings=setup-args="-Ddisable_gds_backend=false" \
        .
    cp "${NIXL_SOURCE_DIR}/build/src/bindings/python/nixl-meta/nixl-${NIXL_VERSION}-py3-none-any.whl" \
        "${NIXL_WHEEL_DIR}/"
    if [[ "${NIXL_INSTALL_PYTHON}" == "true" ]]; then
        PYTHONPATH= python -m pip install --no-deps --no-index --force-reinstall \
            "${NIXL_WHEEL_DIR}/nixl_cu12-${NIXL_VERSION}"-*.whl \
            "${NIXL_WHEEL_DIR}/nixl-${NIXL_VERSION}-py3-none-any.whl"
    fi

    touch "${NIXL_PREFIX}/.psrl-nixl-${NIXL_VERSION}-complete"
}

main() {
    if [[ "${UCX_VERSION}" != "v1.20.0" ]]; then
        echo "The PSRL UCX patch is pinned to UCX v1.20.0." >&2
        exit 1
    fi
    if [[ "${NIXL_VERSION}" != "1.2.0" ]]; then
        echo "The PSRL patch is pinned to NIXL 1.2.0." >&2
        exit 1
    fi
    if [[ "${NIXL_INSTALL_PYTHON}" != "true" && "${NIXL_INSTALL_PYTHON}" != "false" ]]; then
        echo "NIXL_INSTALL_PYTHON must be true or false, got ${NIXL_INSTALL_PYTHON}." >&2
        exit 1
    fi

    require_command git
    require_command make
    require_command meson
    require_command python
    require_command patchelf
    mkdir -p "${BUILD_ROOT}"

    build_ucx
    export PATH="${UCX_PREFIX}/bin:${PATH}"
    export LD_LIBRARY_PATH="${UCX_PREFIX}/lib:${UCX_PREFIX}/lib/ucx:${LD_LIBRARY_PATH:-}"
    export PKG_CONFIG_PATH="${UCX_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
    build_nixl

    echo "Installed UCX ${UCX_VERSION} at ${UCX_PREFIX}."
    echo "Installed NIXL ${NIXL_VERSION} at ${NIXL_PREFIX}."
    echo "Built the NIXL wheels in ${NIXL_WHEEL_DIR}."
}

main "$@"
