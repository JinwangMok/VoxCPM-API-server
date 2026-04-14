#!/usr/bin/env bash
# Build and push the VoxCPM API server image for linux/arm64 (DGX Spark).
#
# Usage:
#   DOCKER_HUB_USER=jinwangmok ./scripts/build-and-push.sh            # build + push
#   PUSH=0 ./scripts/build-and-push.sh                                # local load only (single arch)
#   PLATFORMS=linux/amd64,linux/arm64 ./scripts/build-and-push.sh     # multi-arch
#
# Requirements:
#   - docker with buildx (>= 0.11)
#   - `docker login` completed (or DOCKER_HUB_TOKEN set and we'll log in)
#   - When building arm64 on a non-arm64 host, QEMU must be registered:
#       docker run --privileged --rm tonistiigi/binfmt --install arm64
set -euo pipefail

USER_NAME="${DOCKER_HUB_USER:-jinwangmok}"
IMAGE="${IMAGE:-${USER_NAME}/voxcpm-api-server}"
TAG="${TAG:-latest}"
PLATFORMS="${PLATFORMS:-linux/arm64}"
PUSH="${PUSH:-1}"
CONTEXT_DIR="$(cd "$(dirname "$0")/.." && pwd)/docker/voxcpm"
BUILDER_NAME="${BUILDER_NAME:-voxcpm-builder}"

if [[ ! -f "${CONTEXT_DIR}/Dockerfile" ]]; then
  echo "error: Dockerfile not found at ${CONTEXT_DIR}/Dockerfile" >&2
  exit 1
fi

if [[ -n "${DOCKER_HUB_TOKEN:-}" ]]; then
  echo "${DOCKER_HUB_TOKEN}" | docker login -u "${USER_NAME}" --password-stdin
fi

if ! docker buildx inspect "${BUILDER_NAME}" >/dev/null 2>&1; then
  docker buildx create --name "${BUILDER_NAME}" --driver docker-container --use
else
  docker buildx use "${BUILDER_NAME}"
fi
docker buildx inspect --bootstrap >/dev/null

BUILD_ARGS=(
  buildx build
  --platform "${PLATFORMS}"
  --tag "${IMAGE}:${TAG}"
  --file "${CONTEXT_DIR}/Dockerfile"
)

if [[ "${PUSH}" == "1" ]]; then
  BUILD_ARGS+=(--push)
else
  # --load only supports a single platform.
  if [[ "${PLATFORMS}" == *","* ]]; then
    echo "error: set PUSH=1 or choose a single platform to --load locally" >&2
    exit 1
  fi
  BUILD_ARGS+=(--load)
fi

BUILD_ARGS+=("${CONTEXT_DIR}")

echo "+ docker ${BUILD_ARGS[*]}"
docker "${BUILD_ARGS[@]}"

echo ""
echo "done: ${IMAGE}:${TAG} (${PLATFORMS})"
