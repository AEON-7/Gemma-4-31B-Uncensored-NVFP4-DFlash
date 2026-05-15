<div align="center">

# Gemma 4 31B Uncensored NVFP4 + DFlash

### DGX Spark / GB10 container for Deckard Heretic 31B with official z-lab DFlash drafting

[![Container](https://img.shields.io/badge/ghcr.io-gemma--4--31b--dflash-green?logo=docker)](https://github.com/aeon-7/Gemma-4-31B-Uncensored-NVFP4-DFlash/pkgs/container/gemma-4-31b-uncensored-nvfp4-dflash)
[![Model](https://img.shields.io/badge/HuggingFace-Gemma%204%2031B%20Deckard-yellow?logo=huggingface)](https://huggingface.co/AEON-7/Gemma-4-31B-it-DECKARD-HERETIC-Uncensored-NVFP4)
[![DFlash](https://img.shields.io/badge/DFlash-z--lab%2Fgemma--4--31B--it--DFlash-purple)](https://huggingface.co/z-lab/gemma-4-31B-it-DFlash)
[![License](https://img.shields.io/badge/License-Apache_2.0-green)](LICENSE)
[![☕ Tips](https://img.shields.io/badge/%E2%98%95_Tips-Support_the_work-ff5e5b?style=flat)](https://github.com/AEON-7/AEON-7#-support-the-work)

</div>

This repository packages a validated vLLM container for
`AEON-7/Gemma-4-31B-it-DECKARD-HERETIC-Uncensored-NVFP4` with the official
`z-lab/gemma-4-31B-it-DFlash` drafter. It is built for NVIDIA DGX Spark / GB10
and keeps the useful production surface intact: reasoning, tool calling,
vision/video input, OpenAI-compatible chat completions, and structured output.

## Performance At A Glance

Benchmarked on NVIDIA DGX Spark GB10, SM 12.1, 128 GB unified memory.
The red baseline is the original no-DFlash Deckard NVFP4 run from the model
card. The green result is this DFlash container, warmed, natural prompts,
thinking enabled, 200-token outputs, zero request errors.

| Result | Deployment | c=1 decode | c=16 aggregate | c=32 aggregate | What changed |
|---|---|---:|---:|---:|---|
| 🔴 Before | no speculative decoding | 11 tok/s | 161 tok/s | 162 tok/s | Stock Deckard NVFP4 path. |
| 🟢 After | DFlash k=15 + CUTLASS NVFP4 | 38.82 tok/s avg | 309.74 tok/s avg | 426.66 tok/s avg | 3.5x c=1, 1.9x c=16, 2.6x c=32. |

Peak category wins:

| Category | Warm c=1 decode | Peak c=32 aggregate | Improvement vs 11 tok/s baseline |
|---|---:|---:|---:|
| Coding | 26.81 tok/s | 356.81 tok/s | +144% c=1 |
| Math | 54.02 tok/s | 532.28 tok/s | +391% c=1 |
| Reasoning | 36.74 tok/s | 482.82 tok/s | +234% c=1 |
| Prose | 28.69 tok/s | 295.54 tok/s | +161% c=1 |
| Natural language | 27.49 tok/s | 313.96 tok/s | +150% c=1 |
| Extraction / JSON | 59.16 tok/s | 578.58 tok/s | +438% c=1 |

## Quick Start: DGX Spark / GB10

This path is designed for a fresh DGX Spark where Docker and the NVIDIA
container runtime are already available. The container does not bake the model
weights in; mount the target model and drafter as read-only volumes so updates
are simple and repeatable.

### 1. Prepare directories

Download the target model and drafter on the Spark:

```bash
sudo mkdir -p /models/deckard /models/gemma-dflash
sudo chown -R "$USER:$USER" /models

huggingface-cli download AEON-7/Gemma-4-31B-it-DECKARD-HERETIC-Uncensored-NVFP4 \
  --local-dir /models/deckard

huggingface-cli download z-lab/gemma-4-31B-it-DFlash \
  --local-dir /models/gemma-dflash
```

### 2. Pull the image

```bash
docker pull ghcr.io/aeon-7/gemma-4-31b-uncensored-nvfp4-dflash:latest
```

### 3. Start the server

```bash
docker rm -f gemma31-dflash 2>/dev/null || true

docker run -d --gpus all --ipc=host --network host \
  --name gemma31-dflash \
  --restart unless-stopped \
  -v /models/deckard:/models/deckard:ro \
  -v /models/gemma-dflash:/models/gemma-dflash:ro \
  -e GPU_MEMORY_UTILIZATION=0.82 \
  -e MAX_MODEL_LEN=65536 \
  -e MAX_NUM_SEQS=16 \
  -e MAX_NUM_BATCHED_TOKENS=32768 \
  ghcr.io/aeon-7/gemma-4-31b-uncensored-nvfp4-dflash:latest
```

First boot can take several minutes while vLLM compiles graphs and FlashInfer
autotunes FP4 kernels. Watch startup:

```bash
docker logs -f gemma31-dflash
```

### 4. Verify the endpoint

Health check:

```bash
curl -fsS http://127.0.0.1:8000/health
```

List served model aliases:

```bash
curl -s http://127.0.0.1:8000/v1/models | python3 -m json.tool
```

Smoke-test chat:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deckard-31b",
    "messages": [{"role": "user", "content": "Write one concise paragraph about why speculative decoding helps."}],
    "max_tokens": 600,
    "temperature": 0.2
  }'
```

Endpoint for OpenAI-compatible clients:

```text
http://localhost:8000/v1
```

Served model aliases:

```text
deckard-31b
gemma-4-31b-uncensored
gemma31-dflash
```

For reasoning-heavy prompts, give the model a real output budget. Small
`max_tokens` values can be consumed entirely by the reasoning trace before the
final answer appears.

### 5. Stop or update

```bash
docker pull ghcr.io/aeon-7/gemma-4-31b-uncensored-nvfp4-dflash:latest
docker rm -f gemma31-dflash

docker run -d --gpus all --ipc=host --network host \
  --name gemma31-dflash \
  --restart unless-stopped \
  -v /models/deckard:/models/deckard:ro \
  -v /models/gemma-dflash:/models/gemma-dflash:ro \
  -e GPU_MEMORY_UTILIZATION=0.82 \
  -e MAX_MODEL_LEN=65536 \
  -e MAX_NUM_SEQS=16 \
  -e MAX_NUM_BATCHED_TOKENS=32768 \
  ghcr.io/aeon-7/gemma-4-31b-uncensored-nvfp4-dflash:latest
```

Recreate the container after image updates or runtime flag changes. A plain
`docker start` reuses the old container image and old environment.

## Full DFlash Benchmark

Benchmark command:

```bash
docker run --rm --network host \
  -v /home/albert/stacks/gemma4/bench:/bench \
  ghcr.io/aeon-7/gemma-4-31b-uncensored-nvfp4-dflash:latest \
  bench --base-url http://127.0.0.1:8000/v1 --model deckard-31b \
  --levels 1,4,8,16,32 --max-tokens 200 --temperature 0.0 \
  --runs-per-point 1 --min-samples-per-point 8 --trim-fraction 0.1 \
  --output /bench/gemma31_dflash_category_c1_c32.json
```

Benchmark profile:

```text
MAX_MODEL_LEN=8192
MAX_NUM_SEQS=32
MAX_NUM_BATCHED_TOKENS=32768
GPU_MEMORY_UTILIZATION=0.80
NUM_SPECULATIVE_TOKENS=15
```

| Category | c=1 decode | c=4 aggregate | c=8 aggregate | c=16 aggregate | c=32 aggregate | c=32 TTFT p50 | c=32 TPOT p50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Coding | 26.81 | 91.96 | 174.77 | 251.61 | 356.81 | 1,121 ms | 73.98 ms |
| Math | 54.02 | 134.54 | 242.90 | 342.75 | 532.28 | 1,052 ms | 36.10 ms |
| Reasoning | 36.74 | 127.74 | 230.81 | 340.41 | 482.82 | 1,260 ms | 50.78 ms |
| Prose | 28.69 | 80.39 | 145.90 | 233.55 | 295.54 | 1,042 ms | 91.04 ms |
| Natural language | 27.49 | 88.91 | 151.78 | 237.39 | 313.96 | 1,065 ms | 89.09 ms |
| Extraction / JSON | 59.16 | 166.79 | 290.26 | 452.71 | 578.58 | 854 ms | 38.66 ms |

Raw benchmark exports:

- [`benchmarks-gemma31-dflash-single-stream.json`](benchmarks-gemma31-dflash-single-stream.json)
- [`benchmarks-gemma31-dflash-c1-c32.json`](benchmarks-gemma31-dflash-c1-c32.json)

## What The Container Adds

- vLLM `0.20.1` Gemma 4 DFlash base.
- DFlash support for Gemma 4 using the vLLM PR #41703 lineage.
- Official `z-lab/gemma-4-31B-it-DFlash` drafter support.
- Target attention backend: `triton_attn`.
- Drafter attention backend: `flash_attn`.
- FlashInfer CUTLASS NVFP4 GEMM path on GB10.
- FlashInfer sampler enabled.
- Deckard NVFP4_AWQ ModelOpt loader patch:
  - accepts `quant_algo="NVFP4_AWQ"`,
  - registers and applies `pre_quant_scale`,
  - scrubs rare FP8 NaN block scales produced by ModelOpt 0.42.x.
- Gemma 4 reasoning parser enabled.
- Gemma 4 tool-call parser enabled.
- Multimodal request path preserved.

## Runtime Configuration

The default run profile is intended to be a practical Spark serving profile:

| Variable | Default | Purpose |
|---|---:|---|
| `MAX_MODEL_LEN` | `65536` | Useful long-context serving without filling all memory at boot. |
| `MAX_NUM_SEQS` | `16` | Good mixed-agent capacity for the 31B dense model. |
| `MAX_NUM_BATCHED_TOKENS` | `32768` | Keeps large prefill and agent startup prompts usable. |
| `GPU_MEMORY_UTILIZATION` | `0.82` | Leaves headroom for system services and avoids unified-memory pressure. |
| `NUM_SPECULATIVE_TOKENS` | `15` | DFlash default used for validation. |

For pure short-context benchmarking, `MAX_MODEL_LEN=8192` leaves more room for
parallel decode. For production long-context use, raise or lower the context
limit according to your gateway's real workload.

### Recommended Profiles

Use these as starting points, then validate with your own workload.

| Profile | `MAX_MODEL_LEN` | `MAX_NUM_SEQS` | `MAX_NUM_BATCHED_TOKENS` | `GPU_MEMORY_UTILIZATION` | Best for |
|---|---:|---:|---:|---:|---|
| Balanced Spark default | `65536` | `16` | `32768` | `0.82` | Agent gateways with one large chat plus smaller subagents. |
| Short-context throughput | `8192` | `32` | `32768` | `0.80` | Benchmarking and high-concurrency short tasks. |
| Long-context working chat | `131072` | `4` | `32768` | `0.82` | Fewer sessions with much larger context. |
| Shared Spark services | `32768` | `8` | `32768` | `0.70-0.76` | Leaving room for ASR, TTS, embeddings, or ComfyUI. |

If the system begins touching unified-memory limits, reduce
`MAX_MODEL_LEN`, then `MAX_NUM_SEQS`, before raising
`GPU_MEMORY_UTILIZATION`. Sustained memory pressure usually hurts latency more
than a slightly smaller KV cache.

## API Examples

Chat:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deckard-31b",
    "messages": [{"role": "user", "content": "Write a short scene set on a lunar observatory."}],
    "max_tokens": 1200,
    "temperature": 0.7
  }'
```

Tool calling:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deckard-31b",
    "messages": [{"role": "user", "content": "Use the weather tool for Tokyo in Celsius."}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {"type": "string"},
            "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
          },
          "required": ["city"]
        }
      }
    }],
    "tool_choice": "auto",
    "max_tokens": 1200
  }'
```

## DFlash vs DDTree

Normal decoding verifies one token at a time. DFlash adds a small drafter model
that proposes future tokens, then the full 31B target verifies them in one pass.
When the drafter agrees with the target, the server commits multiple tokens for
roughly the cost of one target step.

DFlash is a flat speculative chain:

```text
prefix -> draft_1 -> draft_2 -> draft_3 -> ...
```

DDTree is the next research step. It spends the same speculative budget on a
tree of alternatives so a rejection near the root does not waste the whole
verification pass:

```text
prefix
  |-- candidate A -> A1 -> A2
  |-- candidate B -> B1
  `-- candidate C -> C1
```

For Gemma 4 dense models, DDTree should be a cleaner target than hybrid models
that require branch-local recurrent-state replay. The intended future scorecard:

| Mode | Status | Role |
|---|---|---|
| 🔴 Baseline | done | Original one-token-at-a-time decode reference. |
| 🟢 DFlash | done | Production speculative decode for this repo. |
| 🟡 DDTree | planned | Experimental tree speculation once quality and fused branch attention are stable. |

## Notes

- First boot is slower because vLLM compiles graphs and FlashInfer autotunes
  CUTLASS FP4 kernels. Subsequent requests on the same container are warm.
- The model may put most early tokens into `message.reasoning`; this is expected
  when the reasoning parser is enabled.
- This image is meant for DGX Spark / GB10. Other Blackwell systems may work,
  but the validation data here is GB10-specific.
