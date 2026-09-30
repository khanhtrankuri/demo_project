import torch

from src.models.backbones import CLIPBackbone, SigLIP2Backbone, SigLIPBackbone


def _lora(last_layers=None):
    result = {"enabled": True, "rank": 2, "alpha": 4, "dropout": 0., "targets": ["q_proj", "v_proj"]}
    if last_layers is not None:
        result["last_layers"] = last_layers
    return result


def test_clip_contract_and_text_lora_are_trainable_only_in_last_layer():
    from transformers import CLIPConfig, CLIPModel
    config = CLIPConfig(
        projection_dim=16,
        text_config=dict(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                         num_attention_heads=2, max_position_embeddings=8, eos_token_id=2,
                         bos_token_id=1, pad_token_id=0),
        vision_config=dict(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                           num_attention_heads=2, image_size=8, patch_size=4))
    backbone = CLIPBackbone(CLIPModel(config), _lora(), _lora(1), gradient_checkpointing=False)
    pixels = torch.randn(2, 3, 8, 8)
    tokens = {"input_ids": torch.tensor([[1, 2, 0], [1, 3, 2]]), "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]])}
    assert backbone.encode_image(pixels).shape == (2, 16)
    assert backbone.get_patch_tokens(pixels).shape == (2, 4, 16)
    assert backbone.encode_text(tokens).shape == (2, 16)
    assert backbone.get_text_tokens(tokens).shape == (2, 3, 16)
    assert len(backbone.text_targets) == 2 and all("layers.1" in name for name in backbone.text_targets)
    trainable = [name for name, parameter in backbone.named_parameters() if parameter.requires_grad]
    assert trainable and all("lora_" in name for name in trainable)


def test_siglip_uses_the_same_contract():
    from transformers import SiglipConfig, SiglipModel
    config = SiglipConfig(
        text_config=dict(vocab_size=32, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=2, num_attention_heads=2, max_position_embeddings=8),
        vision_config=dict(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                           num_attention_heads=2, image_size=8, patch_size=4))
    backbone = SigLIPBackbone(SiglipModel(config), _lora(), _lora(1), gradient_checkpointing=False)
    pixels = torch.randn(2, 3, 8, 8)
    tokens = {"input_ids": torch.tensor([[1, 2, 0], [1, 3, 2]]), "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]])}
    assert backbone.encode_image(pixels).shape == (2, 16)
    assert backbone.get_patch_tokens(pixels).shape == (2, 4, 16)
    assert backbone.encode_text(tokens).shape == (2, 16)
    assert backbone.get_text_tokens(tokens).shape == (2, 3, 16)


def test_siglip2_patchifies_images_behind_the_same_contract():
    from transformers import Siglip2Config, Siglip2Model

    config = Siglip2Config(
        text_config=dict(vocab_size=32, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=2, num_attention_heads=2, max_position_embeddings=8),
        vision_config=dict(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                           num_attention_heads=2, num_patches=4, patch_size=4),
    )
    backbone = SigLIP2Backbone(Siglip2Model(config), _lora(), _lora(1), gradient_checkpointing=False)
    pixels = torch.randn(2, 3, 8, 8)
    tokens = {"input_ids": torch.tensor([[1, 2, 0], [1, 3, 2]]),
              "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]])}
    assert backbone.encode_image(pixels).shape == (2, 16)
    assert backbone.get_patch_tokens(pixels).shape == (2, 4, 16)
    assert backbone.encode_text(tokens).shape == (2, 16)
