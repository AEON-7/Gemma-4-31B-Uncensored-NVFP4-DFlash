#!/usr/bin/env python3
"""Patch vLLM ModelOpt NVFP4 loading for Deckard/Gemma NVFP4_AWQ checkpoints.

The Gemma 4 31B Deckard AWQ_FULL checkpoint was produced with ModelOpt 0.42.x.
It uses quant_algo="NVFP4_AWQ" and stores optional per-linear
`pre_quant_scale` tensors. Older vLLM ModelOpt loaders reject that quant_algo,
do not register `pre_quant_scale`, and do not scrub rare FP8 NaN block scales
emitted by the quantizer.

This is intentionally a small source patch over the DFlash-capable vLLM base.
"""

from __future__ import annotations

from pathlib import Path

import vllm


MARKER = "aeon_gemma31_nvfp4_awq_patch"


def replace_once(text: str, old: str, new: str) -> str:
    if old not in text:
        raise RuntimeError(f"Patch anchor not found:\n{old[:240]}")
    return text.replace(old, new, 1)


def main() -> None:
    root = Path(vllm.__file__).resolve().parent
    path = root / "model_executor/layers/quantization/modelopt.py"
    text = path.read_text()

    if MARKER in text:
        print(f"{MARKER}: already patched")
        return

    text = replace_once(
        text,
        '''from vllm.model_executor.parameter import (
    BlockQuantScaleParameter,''',
        '''from vllm.model_executor.parameter import (
    BasevLLMParameter,
    BlockQuantScaleParameter,''',
    )

    text = replace_once(
        text,
        '''    # FP4
    "NVFP4",
    # MXFP8''',
        '''    # FP4
    "NVFP4",
    # FP4 with AWQ pre-quantization scaling (ModelOpt 0.42.x).
    "NVFP4_AWQ",
    # MXFP8''',
    )

    text = replace_once(
        text,
        '''        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:''',
        f'''        layer.register_parameter("weight_scale", weight_scale)

        # {MARKER}: optional AWQ pre-quantization scale.
        # Present in ModelOpt NVFP4_AWQ checkpoints. Layers without a matching
        # checkpoint tensor keep all-ones scaling and are a no-op.
        pre_quant_scale = BasevLLMParameter(
            data=torch.ones(input_size_per_partition, dtype=torch.bfloat16),
            weight_loader=weight_loader,
        )
        layer.register_parameter("pre_quant_scale", pre_quant_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:''',
    )

    text = replace_once(
        text,
        '''        # Rename ModelOpt checkpoint names to standardized names
        input_global_scale = layer.input_scale.max().to(torch.float32)''',
        f'''        # {MARKER}: scrub FP8 NaN block scales from ModelOpt 0.42.x output.
        # FP8 E4M3 NaN encodings are 0x7F and 0xFF. Setting the block scale to
        # zero is preferable to propagating NaNs through the whole decode.
        ws_raw = layer.weight_scale.data.view(torch.uint8)
        fp8_nan_mask = (ws_raw == 0x7F) | (ws_raw == 0xFF)
        nan_count = int(fp8_nan_mask.sum().item())
        if nan_count:
            ws_raw[fp8_nan_mask] = 0
            logger.warning(
                "Scrubbed %d FP8 NaN values in NVFP4 weight_scale "
                "(shape=%s, total=%d)",
                nan_count,
                list(layer.weight_scale.shape),
                ws_raw.numel(),
            )

        # Rename ModelOpt checkpoint names to standardized names
        input_global_scale = layer.input_scale.max().to(torch.float32)''',
    )

    text = replace_once(
        text,
        '''        # Convert layer to NVFP4 linear kernel format
        self.kernel.process_weights_after_loading(layer)''',
        f'''        # {MARKER}: retain AWQ activation pre-scale for runtime.
        if hasattr(layer, "pre_quant_scale"):
            pqs = layer.pre_quant_scale.data.float()
            if torch.allclose(pqs, torch.ones_like(pqs)):
                layer.has_pre_quant_scale = False
            else:
                layer.has_pre_quant_scale = True
                layer.pre_quant_scale_runtime = Parameter(
                    pqs.to(torch.bfloat16),
                    requires_grad=False,
                )
                logger.info(
                    "Loaded AWQ pre_quant_scale: shape=%s min=%.4f max=%.4f mean=%.4f",
                    list(pqs.shape),
                    pqs.min().item(),
                    pqs.max().item(),
                    pqs.mean().item(),
                )
            del layer.pre_quant_scale
        else:
            layer.has_pre_quant_scale = False

        # Convert layer to NVFP4 linear kernel format
        self.kernel.process_weights_after_loading(layer)''',
    )

    text = replace_once(
        text,
        '''    ) -> torch.Tensor:
        return self.kernel.apply_weights(layer=layer, x=x, bias=bias)''',
        f'''    ) -> torch.Tensor:
        # {MARKER}: apply ModelOpt AWQ activation pre-scale before FP4 matmul.
        if getattr(layer, "has_pre_quant_scale", False):
            x = x * layer.pre_quant_scale_runtime
        return self.kernel.apply_weights(layer=layer, x=x, bias=bias)''',
    )

    path.write_text(text)
    print(f"{MARKER}: patched {path}")


if __name__ == "__main__":
    main()
