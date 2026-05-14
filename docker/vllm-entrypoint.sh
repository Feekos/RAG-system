#!/bin/sh
set -e

unset VLLM_BUILD_COMMIT
unset VLLM_BUILD_PIPELINE
unset VLLM_BUILD_URL
unset VLLM_IMAGE_TAG

exec vllm serve "$@"
