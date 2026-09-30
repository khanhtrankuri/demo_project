"""Hugging Face SigLIP/SigLIP2 adapter implementing the v4 contract."""
import torch
from torch.nn import functional

from .base import VisionLanguageBackbone
from .lora import inject_attention_lora


class SigLIPBackbone(VisionLanguageBackbone):
    def __init__(self, model, vision_lora: dict | None = None, text_lora: dict | None = None,
                 gradient_checkpointing: bool = True):
        super().__init__()
        self.model = model
        self.model.requires_grad_(False)
        vision_width = int(model.config.vision_config.hidden_size)
        text_width = int(model.config.text_config.hidden_size)
        if vision_width != text_width:
            raise ValueError("SigLIP text and vision widths must match for retrieval")
        self.projection_dim = vision_width
        self.vision_targets = inject_attention_lora(model.vision_model, vision_lora) if vision_lora else []
        self.text_targets = inject_attention_lora(
            model.text_model, text_lora, last_layers=text_lora.get("last_layers", 4)
        ) if text_lora and text_lora.get("enabled", True) else []
        if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def encode_image(self, pixel_values):
        return functional.normalize(self.model.get_image_features(pixel_values=pixel_values).float(), dim=-1)

    def get_patch_tokens(self, pixel_values):
        return self.model.vision_model(pixel_values=pixel_values).last_hidden_state.float()

    def encode_text(self, tokens):
        return functional.normalize(self.model.get_text_features(**tokens).float(), dim=-1)

    def get_text_tokens(self, tokens):
        return self.model.text_model(**tokens).last_hidden_state.float()


class SigLIP2Backbone(SigLIPBackbone):
    """SigLIP2 adapter that patchifies square crop tensors for the NaFlex API."""

    def _vision_inputs(self, pixel_values):
        patch = int(self.model.config.vision_config.patch_size)
        height, width = pixel_values.shape[-2:]
        if height % patch or width % patch:
            raise ValueError("SigLIP2 input dimensions must be divisible by patch_size")
        patches = pixel_values.unfold(2, patch, patch).unfold(3, patch, patch)
        patches = patches.permute(0, 2, 3, 1, 4, 5).flatten(3).flatten(1, 2)
        spatial_shapes = torch.tensor([height // patch, width // patch], device=pixel_values.device).repeat(len(pixel_values), 1)
        attention_mask = torch.ones(patches.shape[:2], dtype=torch.bool, device=pixel_values.device)
        return {"pixel_values": patches, "pixel_attention_mask": attention_mask,
                "spatial_shapes": spatial_shapes}

    def encode_image(self, pixel_values):
        return functional.normalize(self.model.get_image_features(**self._vision_inputs(pixel_values)).float(), dim=-1)

    def get_patch_tokens(self, pixel_values):
        inputs = self._vision_inputs(pixel_values)
        return self.model.vision_model(
            pixel_values=inputs["pixel_values"], attention_mask=inputs["pixel_attention_mask"],
            spatial_shapes=inputs["spatial_shapes"],
        ).last_hidden_state.float()
