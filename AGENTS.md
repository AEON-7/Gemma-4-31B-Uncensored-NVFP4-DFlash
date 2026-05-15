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

## Production Reminder

DFlash is the production target for this repo. DDTree is a follow-up research
track and should be packaged separately after it produces quality-stable output.
