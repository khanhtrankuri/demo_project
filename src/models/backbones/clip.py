"""Hugging Face CLIP adapter implementing the v4 backbone contract."""
from torch.nn import functional

from .base import VisionLanguageBackbone
from .lora import inject_attention_lora


class CLIPBackbone(VisionLanguageBackbone):
    def __init__(self, model, vision_lora: dict | None = None, text_lora: dict | None = None,
                 gradient_checkpointing: bool = True):
        super().__init__()
        self.model = model
        self.model.requires_grad_(False)
        self.projection_dim = int(model.config.projection_dim)
        self.vision_targets = inject_attention_lora(model.vision_model, vision_lora) if vision_lora else []
        self.text_targets = inject_attention_lora(
            model.text_model, text_lora, last_layers=text_lora.get("last_layers", 4)
        ) if text_lora and text_lora.get("enabled", True) else []
        if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def encode_image(self, pixel_values):
        pooled = self.model.vision_model(pixel_values=pixel_values).pooler_output
        return functional.normalize(self.model.visual_projection(pooled).float(), dim=-1)

    def get_patch_tokens(self, pixel_values):
        hidden = self.model.vision_model(pixel_values=pixel_values).last_hidden_state[:, 1:]
        return self.model.visual_projection(hidden).float()

    def encode_text(self, tokens):
        return functional.normalize(self.model.get_text_features(**tokens).float(), dim=-1)

    def get_text_tokens(self, tokens):
        hidden = self.model.text_model(**tokens).last_hidden_state
        return self.model.text_projection(hidden).float()
