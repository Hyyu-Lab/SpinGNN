#!/usr/bin/env bash
set -Eeuo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WORKSPACE=$(cd "$REPO/.." && pwd)
TE3NN="$WORKSPACE/validation/vendor/T-e3nn"
IMAGE=${1:-spingnn:cuda128}
EXPECTED_TE3NN=75334027fffc38f2268db8ab1eed281a18ffdbf2
BUILDER=${SPINGNN_CONTAINER_BUILDER:-docker}
OFFLINE=${SPINGNN_OFFLINE:-0}

case "$BUILDER" in
    docker)
        command -v docker >/dev/null
        docker version >/dev/null
        ;;
    buildah)
        command -v buildah >/dev/null
        ;;
    *)
        echo "SPINGNN_CONTAINER_BUILDER must be docker or buildah" >&2
        exit 2
        ;;
esac
test -d "$TE3NN/.git"

if [[ -n $(git -C "$REPO" status --porcelain) ]]; then
    echo "SpinGNN worktree must be clean so the image source is auditable" >&2
    exit 2
fi
if [[ -n $(git -C "$TE3NN" status --porcelain) ]]; then
    echo "T-e3nn worktree must be clean so the image source is auditable" >&2
    exit 2
fi

SPINGNN_COMMIT=$(git -C "$REPO" rev-parse HEAD)
TE3NN_COMMIT=$(git -C "$TE3NN" rev-parse HEAD)
if [[ "$TE3NN_COMMIT" != "$EXPECTED_TE3NN" ]]; then
    echo "Expected T-e3nn $EXPECTED_TE3NN, found $TE3NN_COMMIT" >&2
    exit 2
fi

CONTEXT=$(mktemp -d "${TMPDIR:-/tmp}/spingnn-docker-context.XXXXXX")
cleanup() {
    case "$CONTEXT" in
        "${TMPDIR:-/tmp}"/spingnn-docker-context.*) rm -rf -- "$CONTEXT" ;;
        *) echo "Refusing to remove unexpected context path: $CONTEXT" >&2 ;;
    esac
}
trap cleanup EXIT

mkdir -p "$CONTEXT/SpinGNN" "$CONTEXT/T-e3nn" "$CONTEXT/python" "$CONTEXT/wheelhouse"
git -C "$REPO" archive --format=tar HEAD | tar -xf - -C "$CONTEXT/SpinGNN"
git -C "$TE3NN" archive --format=tar HEAD | tar -xf - -C "$CONTEXT/T-e3nn"
cp "$REPO/docker/.dockerignore" "$CONTEXT/.dockerignore"

if [[ "$OFFLINE" == 1 ]]; then
    OFFLINE_ASSETS=${SPINGNN_OFFLINE_ASSETS:-}
    test -d "$OFFLINE_ASSETS/python/cpython-3.10.20-linux-x86_64-gnu"
    test -d "$OFFLINE_ASSETS/wheelhouse"
    cp -a "$OFFLINE_ASSETS/python/." "$CONTEXT/python/"
    cp -a "$OFFLINE_ASSETS/wheelhouse/." "$CONTEXT/wheelhouse/"
fi

echo "SPINGNN_IMAGE_BUILD_START builder=$BUILDER image=$IMAGE source=$SPINGNN_COMMIT te3nn=$TE3NN_COMMIT"
if [[ "$BUILDER" == docker ]]; then
    docker build \
        --build-arg "SPINGNN_SOURCE_COMMIT=$SPINGNN_COMMIT" \
        --build-arg "TE3NN_SOURCE_COMMIT=$TE3NN_COMMIT" \
        --build-arg "SPINGNN_OFFLINE=$OFFLINE" \
        --file "$CONTEXT/SpinGNN/docker/Dockerfile" \
        --tag "$IMAGE" \
        "$CONTEXT"
    docker image inspect "$IMAGE" --format '{{json .RepoTags}} {{.Id}} {{.Size}}'
else
    buildah --storage-driver "${BUILDAH_STORAGE_DRIVER:-vfs}" bud \
        --isolation "${BUILDAH_ISOLATION:-chroot}" \
        --layers \
        --format docker \
        --tls-verify="${BUILDAH_TLS_VERIFY:-true}" \
        --build-arg "SPINGNN_SOURCE_COMMIT=$SPINGNN_COMMIT" \
        --build-arg "TE3NN_SOURCE_COMMIT=$TE3NN_COMMIT" \
        --build-arg "SPINGNN_OFFLINE=$OFFLINE" \
        --file "$CONTEXT/SpinGNN/docker/Dockerfile" \
        --tag "$IMAGE" \
        "$CONTEXT"
    buildah --storage-driver "${BUILDAH_STORAGE_DRIVER:-vfs}" inspect --type image "$IMAGE"
fi
echo "SPINGNN_IMAGE_BUILD_OK builder=$BUILDER image=$IMAGE source=$SPINGNN_COMMIT"
