"""Numerically stable multi-positive contrastive objectives."""
import torch
from torch.nn import functional


def positive_mask_from_ids(image_ids, text_ids, device=None) -> torch.Tensor:
    return torch.tensor([[left == right for right in text_ids] for left in image_ids],
                        dtype=torch.bool, device=device)


def _directional(logits: torch.Tensor, positives: torch.Tensor) -> torch.Tensor:
    valid = positives.any(dim=1)
    if not valid.any():
        return logits.sum() * 0
    numerator = torch.logsumexp(logits.masked_fill(~positives, -torch.inf), dim=1)
    denominator = torch.logsumexp(logits, dim=1)
    return (denominator[valid] - numerator[valid]).mean()


def multi_positive_info_nce(images: torch.Tensor, texts: torch.Tensor, positives: torch.Tensor,
                            logit_scale: torch.Tensor | float,
                            image_queue: torch.Tensor | None = None,
                            text_queue: torch.Tensor | None = None,
                            queued_image_positives: torch.Tensor | None = None,
                            queued_text_positives: torch.Tensor | None = None) -> torch.Tensor:
    """Symmetric InfoNCE with arbitrary many-to-many positives and queues."""
    if positives.shape != (len(images), len(texts)):
        raise ValueError("positive mask shape must be [images,texts]")
    images, texts = functional.normalize(images.float(), dim=-1), functional.normalize(texts.float(), dim=-1)
    scale = torch.as_tensor(logit_scale, device=images.device, dtype=torch.float32).clamp(max=100)
    text_gallery = texts if text_queue is None else torch.cat([texts, functional.normalize(text_queue.detach().float(), dim=-1)], 0)
    image_gallery = images if image_queue is None else torch.cat([images, functional.normalize(image_queue.detach().float(), dim=-1)], 0)
    i2t_positive = positives
    if text_queue is not None:
        extra = (torch.zeros(len(images), len(text_queue), dtype=torch.bool, device=images.device)
                 if queued_text_positives is None else queued_text_positives.to(images.device))
        i2t_positive = torch.cat([positives, extra], 1)
    t2i_positive = positives.T
    if image_queue is not None:
        extra = (torch.zeros(len(texts), len(image_queue), dtype=torch.bool, device=images.device)
                 if queued_image_positives is None else queued_image_positives.to(images.device))
        t2i_positive = torch.cat([positives.T, extra], 1)
    return (_directional(scale * images @ text_gallery.T, i2t_positive)
            + _directional(scale * texts @ image_gallery.T, t2i_positive)) / 2
