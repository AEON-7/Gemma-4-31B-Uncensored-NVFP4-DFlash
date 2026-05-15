# Agent Notes

This repository packages Gemma 4 31B Deckard Heretic Uncensored NVFP4 with
official z-lab DFlash drafting for DGX Spark / GB10.

## Priorities

- Preserve the target model capabilities: thinking/reasoning, tool calling,
  multimodal input, JSON/structured output, and OpenAI-compatible serving.
- Keep the ModelOpt NVFP4_AWQ patch focused and easy to audit.
- Benchmark with natural prompts, not synthetic random text.
- Capture TTFT, TPOT, median decode, peak decode, prompt processing, aggregate
  throughput, and error counts.
- Keep the red/green benchmark contrast in the README current whenever runtime
  flags or base images change.
- Do not claim DDTree results from this repo until the DDTree-specific package
  exists and has its own benchmark pass.

## Runtime Shape

- Target model: `AEON-7/Gemma-4-31B-it-DECKARD-HERETIC-Uncensored-NVFP4`
- Drafter: `z-lab/gemma-4-31B-it-DFlash`
- Target attention backend: `triton_attn`
- Drafter attention backend: `flash_attn`
- Quantization: `modelopt`
- KV cache: `auto`
- Default Spark profile: `MAX_MODEL_LEN=65536`, `MAX_NUM_SEQS=16`,
  `MAX_NUM_BATCHED_TOKENS=32768`, `GPU_MEMORY_UTILIZATION=0.82`
- Published image: `ghcr.io/aeon-7/gemma-4-31b-uncensored-nvfp4-dflash`
- Current validated tags: `latest`, `v1`
- Benchmark profile used for README tables: `MAX_MODEL_LEN=8192`,
  `MAX_NUM_SEQS=32`, `MAX_NUM_BATCHED_TOKENS=32768`,
  `GPU_MEMORY_UTILIZATION=0.80`

## Deployment Guide For Agents

When asked to deploy this package on a local machine, follow this sequence.

1. Confirm the machine is actually a DGX Spark / GB10 or a compatible
   Linux/arm64 NVIDIA Blackwell system. This container is published for
   `linux/arm64`; do not assume it will run on x86 workstations.
2. Stop other large GPU containers before first launch. vLLM graph compilation,
   FP4 autotuning, and model loading need clean memory headroom.
3. Confirm the model directories exist and are populated:
   - Target: `/models/deckard`
   - Drafter: `/models/gemma-dflash`
4. Pull `ghcr.io/aeon-7/gemma-4-31b-uncensored-nvfp4-dflash:latest`.
5. Choose a runtime profile from the table below.
6. Start the container with `--ipc=host`, `--network host`, `--gpus all`, and
   read-only model mounts.
7. Wait for `/health` to respond before configuring a gateway.
8. Smoke-test `/v1/models`, normal chat, and tool calling.
9. If benchmarking, use natural prompts and capture TTFT, TPOT, prompt
   processing, median decode, peak decode, aggregate throughput, and errors.

### Local Machine Inspection

Before choosing flags, collect:

```bash
uname -m
free -h
docker ps --format '{{.Names}} {{.Image}} {{.Status}}'
docker images ghcr.io/aeon-7/gemma-4-31b-uncensored-nvfp4-dflash
nvidia-smi || true
```

Interpretation:

- `uname -m` should be `aarch64` for the published Spark image.
- If other vLLM, ASR, TTS, embedding, or ComfyUI containers are running, use the
  shared-services profile or stop them before benchmarking.
- If `nvidia-smi` reports `N/A` memory on unified-memory Spark systems, use
  `free -h`, vLLM startup logs, and observed swap/unified-memory pressure.
- If the machine is hot or already at sustained high utilization, let it cool
  before comparing benchmark numbers.

### Runtime Profiles

Use the smallest profile that preserves the user's requested capability. Lower
context and sequence capacity usually improves latency because less memory is
reserved for KV cache.

| Local goal | `MAX_MODEL_LEN` | `MAX_NUM_SEQS` | `MAX_NUM_BATCHED_TOKENS` | `GPU_MEMORY_UTILIZATION` | Notes |
|---|---:|---:|---:|---:|---|
| Balanced Spark default | `65536` | `16` | `32768` | `0.82` | Best first choice for local agent gateways. |
| Maximum short-task throughput | `8192` | `32` | `32768` | `0.80` | Matches the published benchmark profile. |
| Long-context primary chat | `131072` | `4` | `32768` | `0.82` | Use when a single working chat needs large context. |
| Shared Spark with ASR/TTS/embeddings | `32768` | `8` | `32768` | `0.70-0.76` | Leaves memory for nearby services. |
| Debug / smoke test | `8192` | `4` | `8192` | `0.75-0.78` | Fastest safe profile for boot validation. |

Profile selection:

- Use balanced default when the user wants a general local agent server.
- Use short-task throughput when the user asks for benchmark numbers or mostly
  short coding/tool-call bursts.
- Use long-context primary chat only when the user explicitly values context
  capacity over concurrency and decode latency.
- Use shared Spark when voice, embeddings, image generation, or other local GPU
  services must remain online.
- Use debug/smoke first after a fresh build, then relaunch with the real profile.

### Launch Template

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
  -e NUM_SPECULATIVE_TOKENS=15 \
  ghcr.io/aeon-7/gemma-4-31b-uncensored-nvfp4-dflash:latest
```

### Verification Commands

```bash
docker logs -f gemma31-dflash
curl -fsS http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/v1/models | python3 -m json.tool
```

Chat smoke test:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deckard-31b",
    "messages": [{"role": "user", "content": "Write one concise paragraph about DFlash."}],
    "max_tokens": 600,
    "temperature": 0.2
  }'
```

Tool-call smoke test:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
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
    "max_tokens": 600
  }'
```

### Gateway Settings

- Base URL: `http://<host-ip>:8000/v1`
- API key: any non-empty local token unless a proxy enforces auth.
- Model: `deckard-31b`
- Tool calling: enabled.
- Reasoning: enabled; preserve `message.reasoning` or `message.reasoning_content`
  when the client exposes it.
- Vision/video input: enabled.
- Default output budget: use at least `1200` tokens for thinking tasks.

### Tuning Rules

- If TTFT is high but decode is good, check prompt size and prefix-cache hit
  behavior before changing DFlash.
- If decode slows badly under concurrency, lower `MAX_MODEL_LEN` first. That
  frees KV headroom without sacrificing tool calling or reasoning capability.
- If the machine also runs ASR, TTS, embeddings, or image generation, start with
  `GPU_MEMORY_UTILIZATION=0.70-0.76`.
- Keep `NUM_SPECULATIVE_TOKENS=15` unless a benchmark on this exact model and
  workload proves a better value.
- Do not disable the Gemma reasoning or tool parsers; that would reduce the
  advertised capability surface.

## Production Reminder

DFlash is the production target for this repo. DDTree is a follow-up research
track and should be packaged separately after it produces quality-stable output.
