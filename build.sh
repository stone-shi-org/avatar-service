#!/usr/bin/env bash
# Build (and optionally test / push) the image.
#
#   ./build.sh                          build avatar:latest locally
#   ./build.sh --test                   ... then run the GPU-free test suite inside it (JUnit -> test-reports/)
#   ./build.sh -p -r registry/ns        tag as registry/ns/avatar:{latest,<sha>} and push
#   AVATAR_REGISTRY=registry/ns ./build.sh -p
#
# No registry is hardcoded: this repo is public.
set -euo pipefail
cd "$(dirname "$0")"

REGISTRY="${AVATAR_REGISTRY:-}"
IMAGE=avatar
PUSH=0
TEST=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--push) PUSH=1; shift ;;
    -r|--registry) REGISTRY="$2"; shift 2 ;;
    --test) TEST=1; shift ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done
if [[ $PUSH -eq 1 && -z "$REGISTRY" ]]; then
  echo "--push needs -r <registry> or AVATAR_REGISTRY" >&2
  exit 1
fi

REPO="${REGISTRY:+${REGISTRY}/}${IMAGE}"
SHA="$(git rev-parse --short=7 HEAD 2>/dev/null || echo dev)"
if [[ -n "$(git status --porcelain 2>/dev/null)" ]]; then SHA="${SHA}-dirty"; fi

echo "==> building ${REPO}:latest (${SHA})"
DOCKER_BUILDKIT=1 docker build --pull --build-arg GIT_SHA="$SHA" -t "${REPO}:latest" -t "${REPO}:${SHA}" .

if [[ $TEST -eq 1 ]]; then
  echo "==> tests (fake renderer, no GPU)"
  name="avatar-test-$$"
  rm -rf test-reports && mkdir -p test-reports
  set +e
  docker run --name "$name" --entrypoint python "${REPO}:latest" \
    -m pytest -q -p no:cacheprovider -W ignore::DeprecationWarning tests --junitxml=/tmp/junit.xml
  rc=$?
  set -e
  docker cp "$name:/tmp/junit.xml" test-reports/junit.xml || true
  docker rm -f "$name" >/dev/null
  [[ $rc -eq 0 ]] || { echo "tests failed" >&2; exit $rc; }
fi

if [[ $PUSH -eq 1 ]]; then
  echo "==> pushing ${REPO}:latest and ${REPO}:${SHA}"
  docker push "${REPO}:${SHA}"
  docker push "${REPO}:latest"
fi
