"""Exact logical-batch feature gradients with micro-batch encoder replay."""
from contextlib import nullcontext

import torch


def precision_context(device, precision):
    if str(device).startswith("cuda") and precision == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def cached_backward(encode, images, mask, loss_fn, micro_batch_size, device="cpu", precision="fp32"):
    """Backpropagate one full contrastive loss, keeping one encoder graph at a time.

    Inputs stay on CPU between passes. RNG replay keeps LoRA dropout identical.
    loss_fn may include trainable fusion heads; its gradients are computed once.
    Caller owns zero_grad/clip/step. No averaging over micro-batches is needed.
    """
    if micro_batch_size < 1 or not len(images):
        raise ValueError("Nonempty batch and positive micro_batch_size required")
    use_cuda = str(device).startswith("cuda")
    cuda_device = torch.device(device).index if use_cuda else None
    if use_cuda and cuda_device is None:
        cuda_device = torch.cuda.current_device()
    states, chunks = [], []
    for start in range(0, len(images), micro_batch_size):
        states.append((torch.get_rng_state(), torch.cuda.get_rng_state(cuda_device) if use_cuda else None))
        with torch.no_grad(), precision_context(device, precision):
            chunks.append(encode(images[start:start+micro_batch_size], mask[start:start+micro_batch_size]))
    leaf = torch.cat(chunks).detach().requires_grad_(True)
    if not torch.isfinite(leaf).all():
        raise ValueError("Non-finite encoder features")
    # Small retrieval heads/loss use FP32, independent of the BF16 backbone.
    loss = loss_fn(leaf)
    if not torch.isfinite(loss):
        raise ValueError("Non-finite training loss")
    loss.backward()
    gradients = leaf.grad.detach()
    for i, start in enumerate(range(0, len(images), micro_batch_size)):
        cpu_rng, cuda_rng = states[i]
        with torch.random.fork_rng(devices=[cuda_device] if use_cuda else []):
            torch.set_rng_state(cpu_rng)
            if use_cuda:
                torch.cuda.set_rng_state(cuda_rng, cuda_device)
            with precision_context(device, precision):
                replay = encode(images[start:start+micro_batch_size], mask[start:start+micro_batch_size])
            replay.backward(gradients[start:start+micro_batch_size])
    return float(loss.detach())
