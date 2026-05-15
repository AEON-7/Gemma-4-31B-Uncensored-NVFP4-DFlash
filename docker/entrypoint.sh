#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-serve}"
if [[ $# -gt 0 ]]; then
  shift
fi

ROOT="/opt/gemma31-dflash"

case "$MODE" in
  serve|dflash)
    MODEL_DIR="${MODEL_DIR:-/models/deckard}"
    DFLASH_DIR="${DFLASH_DIR:-/models/gemma-dflash}"
    SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deckard-31b gemma-4-31b-uncensored gemma31-dflash}"
    PORT="${PORT:-8000}"
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
    MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
    GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.82}"
    NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-15}"
    ATTENTION_BACKEND="${ATTENTION_BACKEND:-triton_attn}"
    DFLASH_ATTENTION_BACKEND="${DFLASH_ATTENTION_BACKEND:-flash_attn}"
    COMPILATION_CONFIG="${COMPILATION_CONFIG:-{\"cudagraph_capture_sizes\":[1,2,4,8,12,16,20,24,28,32,40,48,56,64]}}"

    export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1a}"
    export TORCH_MATMUL_PRECISION="${TORCH_MATMUL_PRECISION:-high}"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    export NVIDIA_FORWARD_COMPAT="${NVIDIA_FORWARD_COMPAT:-1}"
    export NVIDIA_DISABLE_REQUIRE="${NVIDIA_DISABLE_REQUIRE:-1}"
    export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-1}"

    SPEC_CONFIG=$(printf '{"method":"dflash","model":"%s","num_speculative_tokens":%s,"attention_backend":"%s"}' \
      "$DFLASH_DIR" "$NUM_SPECULATIVE_TOKENS" "$DFLASH_ATTENTION_BACKEND")

    exec vllm serve "$MODEL_DIR" \
      --served-model-name $SERVED_MODEL_NAME \
      --host 0.0.0.0 \
      --port "$PORT" \
      --tensor-parallel-size 1 \
      --dtype auto \
      --quantization modelopt \
      --kv-cache-dtype auto \
      --max-model-len "$MAX_MODEL_LEN" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --enable-chunked-prefill \
      --enable-prefix-caching \
      --load-format safetensors \
      --trust-remote-code \
      --enable-auto-tool-choice \
      --tool-call-parser gemma4 \
      --reasoning-parser gemma4 \
      --attention-backend "$ATTENTION_BACKEND" \
      --generation-config vllm \
      --compilation-config "$COMPILATION_CONFIG" \
      --limit-mm-per-prompt '{"image": 4, "video": 2}' \
      --mm-encoder-tp-mode data \
      --mm-processor-cache-type shm \
      --mm-shm-cache-max-object-size-mb 256 \
      --speculative-config "$SPEC_CONFIG" \
      ${EXTRA_VLLM_ARGS:-}
    ;;
  baseline)
    MODEL_DIR="${MODEL_DIR:-/models/deckard}"
    SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deckard-31b-baseline}"
    PORT="${PORT:-8000}"
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
    MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
    GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.82}"
    ATTENTION_BACKEND="${ATTENTION_BACKEND:-triton_attn}"

    export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1a}"
    export TORCH_MATMUL_PRECISION="${TORCH_MATMUL_PRECISION:-high}"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    export NVIDIA_FORWARD_COMPAT="${NVIDIA_FORWARD_COMPAT:-1}"
    export NVIDIA_DISABLE_REQUIRE="${NVIDIA_DISABLE_REQUIRE:-1}"

    exec vllm serve "$MODEL_DIR" \
      --served-model-name "$SERVED_MODEL_NAME" \
      --host 0.0.0.0 \
      --port "$PORT" \
      --tensor-parallel-size 1 \
      --dtype auto \
      --quantization modelopt \
      --kv-cache-dtype auto \
      --max-model-len "$MAX_MODEL_LEN" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --enable-chunked-prefill \
      --enable-prefix-caching \
      --load-format safetensors \
      --trust-remote-code \
      --enable-auto-tool-choice \
      --tool-call-parser gemma4 \
      --reasoning-parser gemma4 \
      --attention-backend "$ATTENTION_BACKEND" \
      --generation-config vllm \
      --limit-mm-per-prompt '{"image": 4, "video": 2}' \
      --mm-encoder-tp-mode data \
      --mm-processor-cache-type shm \
      --mm-shm-cache-max-object-size-mb 256 \
      ${EXTRA_VLLM_ARGS:-}
    ;;
  bench)
    exec python3 "$ROOT/scripts/bench_categories_stream.py" "$@"
    ;;
  bash|shell)
    exec bash "$@"
    ;;
  *)
    exec "$MODE" "$@"
    ;;
esac
