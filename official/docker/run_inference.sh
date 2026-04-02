#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
IMAGE_NAME=${SPOT_RNA_OFFICIAL_IMAGE:-spot-rna-official}

usage() {
    cat <<'EOF'
Usage: official/docker/run_inference.sh <repo-input-path> <repo-output-dir> [SPOT-RNA args...]

Examples:
  official/docker/run_inference.sh sample_inputs/single_seq.fasta outputs/ --cpu 32
  official/docker/run_inference.sh sample_inputs/batch_seq.fasta outputs/ --gpu 0

Paths must be relative to the repository root or already live under it, because the
wrapper mounts only this repository into the container.

Build the image first:
  docker build -f official/docker/Dockerfile -t spot-rna-official .
EOF
}

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ] || [ "$#" -lt 2 ]; then
    usage
    exit 0
fi

INPUT_PATH=$1
OUTPUT_PATH=$2
shift 2

rewrite_repo_path() {
    case "$1" in
        "$REPO_ROOT")
            printf '/workspace\n'
            ;;
        "$REPO_ROOT"/*)
            printf '/workspace/%s\n' "${1#"$REPO_ROOT"/}"
            ;;
        *)
            printf '%s\n' "$1"
            ;;
    esac
}

case "$INPUT_PATH" in
    "$REPO_ROOT"/*) ;;
    /*)
        printf '%s\n' "Input path must be inside the repository: $INPUT_PATH" >&2
        exit 1
        ;;
esac

case "$OUTPUT_PATH" in
    "$REPO_ROOT"/*) ;;
    /*)
        printf '%s\n' "Output path must be inside the repository: $OUTPUT_PATH" >&2
        exit 1
        ;;
esac

INPUT_PATH=$(rewrite_repo_path "$INPUT_PATH")
OUTPUT_PATH=$(rewrite_repo_path "$OUTPUT_PATH")

docker run --rm \
    -v "$REPO_ROOT:/workspace" \
    -w /workspace \
    "$IMAGE_NAME" \
    --inputs "$INPUT_PATH" \
    --outputs "$OUTPUT_PATH" \
    "$@"
