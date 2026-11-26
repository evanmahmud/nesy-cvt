from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


FLOP_COUNT_METHOD = (
    "Analytical estimate: Conv2d, Linear, multi-head/window-attention matrix operations, "
    "cosine-classifier matrix multiplication, and approximate 2D FFT work. Uses 2 FLOPs/MAC; "
    "excludes normalization, activation, softmax, pooling, and most elementwise operations."
)


def estimate_flops_per_image(model, *, device, image_size: int, in_channels: int = 3) -> dict[str, Any]:
    """Estimate inference FLOPs for one image using explicit module hooks.

    This is a transparent analytical estimate, not a hardware instruction count. FLOPs for
    common convolution/linear/attention operations are counted; FFT arithmetic is estimated
    using the common O(N log N) approximation.
    """
    import torch
    from torch import nn

    from .model import CosinePrototypeClassifier, SpectralMixer

    counts: dict[str, float] = defaultdict(float)
    hooks = []

    def conv2d_hook(module, _inputs, output):
        if isinstance(output, torch.Tensor):
            kernel_h, kernel_w = module.kernel_size
            macs = (
                output.numel()
                * (module.in_channels // module.groups)
                * kernel_h
                * kernel_w
            )
            counts["conv2d"] += 2.0 * macs

    def linear_hook(module, _inputs, output):
        if isinstance(output, torch.Tensor):
            counts["linear"] += 2.0 * output.numel() * module.in_features

    def mha_hook(module, inputs, _output):
        query, key, value = inputs[:3]
        if module.batch_first:
            batch_size, query_length, embed_dim = query.shape
            key_length = key.shape[1]
            value_length = value.shape[1]
        else:
            query_length, batch_size, embed_dim = query.shape
            key_length = key.shape[0]
            value_length = value.shape[0]
        heads = module.num_heads
        head_dim = embed_dim // heads
        projection_flops = (
            2.0
            * batch_size
            * embed_dim
            * embed_dim
            * (query_length + key_length + value_length + query_length)
        )
        attention_flops = 2.0 * batch_size * heads * head_dim * (
            query_length * key_length + query_length * value_length
        )
        counts["multihead_attention"] += projection_flops + attention_flops

    def cosine_classifier_hook(module, inputs, _output):
        features = inputs[0]
        counts["cosine_classifier"] += (
            2.0 * features.shape[0] * module.prototypes.shape[0] * features.shape[-1]
        )

    def spectral_mixer_hook(_module, inputs, _output):
        pixels = inputs[0]
        batch_size, channels, height, width = pixels.shape
        # A 2-D forward and inverse real FFT pair is approximated as 5*N*log2(N)
        # operations per channel. Pointwise complex gate multiplication is also included.
        fft_flops = (
            5.0
            * batch_size
            * channels
            * height
            * width
            * (math.log2(height) + math.log2(width))
        )
        frequency_bins = height * (width // 2 + 1)
        gate_flops = 6.0 * batch_size * channels * frequency_bins
        counts["spectral_fft_estimate"] += fft_flops + gate_flops

    def window_attention_hook(module, inputs, _output):
        pixels = inputs[0]
        if pixels.ndim != 4:
            return
        batch_size, height, width, _channels = pixels.shape
        window_h, window_w = module.window_size
        heads = module.num_heads
        head_dim = module.qkv.in_features // heads
        windows_per_image = math.ceil(height / window_h) * math.ceil(width / window_w)
        tokens_per_window = window_h * window_w
        counts["window_attention"] += (
            4.0
            * batch_size
            * windows_per_image
            * heads
            * tokens_per_window
            * tokens_per_window
            * head_dim
        )

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv2d_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))
        elif isinstance(module, nn.MultiheadAttention):
            hooks.append(module.register_forward_hook(mha_hook))
        elif isinstance(module, CosinePrototypeClassifier):
            hooks.append(module.register_forward_hook(cosine_classifier_hook))
        elif isinstance(module, SpectralMixer):
            hooks.append(module.register_forward_hook(spectral_mixer_hook))
        elif module.__class__.__name__.startswith("ShiftedWindowAttention"):
            hooks.append(module.register_forward_hook(window_attention_hook))

    was_training = model.training
    model.eval()
    dummy = torch.zeros((1, in_channels, image_size, image_size), device=device)
    try:
        with torch.inference_mode():
            model(dummy)
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)

    total_flops = sum(counts.values())
    return {
        "flops_per_image": int(round(total_flops)),
        "flops_g_per_image": total_flops / 1_000_000_000.0,
        "flops_breakdown": {name: int(round(value)) for name, value in sorted(counts.items())},
        "flops_method": FLOP_COUNT_METHOD,
    }
