
import torch
from PIL import Image
from torch import nn

from src.models.multicamera import CameraTransformerFusion, GeometryEncoder, MultiScaleTransform, SpatialEncoder


class TinyBackbone(nn.Module):
    projection_dim = 16

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, self.projection_dim)

    def encode_image(self, pixels):
        return self.projection(pixels.mean((-1, -2)))

    def get_patch_tokens(self, pixels):
        pooled = pixels.unfold(2, 4, 4).unfold(3, 4, 4).mean((-1, -2)).flatten(2).transpose(1, 2)
        return self.projection(pooled)


def test_multiscale_transform_preserves_global_and_horizontal_detail():
    image = Image.new("RGB", (32, 8), (0, 0, 0))
    for x in range(0, 8):
        for y in range(8):
            image.putpixel((x, y), (255, 0, 0))
    transform = MultiScaleTransform(8, [.5] * 3, [.5] * 3, 3)
    result = transform(image)
    assert result.shape == (4, 3, 8, 8)
    assert result[1, 0].mean() > .9  # left crop keeps the small red region large
    assert result[3, 0].mean() < -.9


def test_spatial_and_camera_transformers_mask_missing_views_and_backpropagate():
    spatial = SpatialEncoder(TinyBackbone(), 16, num_spatial_crops=3, heads=4)
    images = torch.randn(2, 6, 4, 3, 8, 8)
    mask = torch.tensor([[True] * 6, [True, False, False, False, False, False]])
    images[1, 1:] = float("nan")
    encoded = spatial(images, mask, return_patches=True)
    assert encoded.camera_tokens.shape == (2, 6, 16)
    assert encoded.patch_tokens.shape[:3] == (2, 6, 4)
    assert torch.isfinite(encoded.camera_tokens).all()
    geometry = GeometryEncoder(16)(torch.zeros(2, 6, 4))
    fusion = CameraTransformerFusion(16, layers=2, heads=4).eval()
    result = fusion(encoded.camera_tokens, mask, geometry)
    assert result["scene"].shape == (2, 16)
    assert result["cameras"][1, 1:].abs().sum() == 0
    result["scene"].sum().backward()
    assert fusion.scene_cls.grad is not None


def test_camera_dropout_never_removes_every_available_camera():
    fusion = CameraTransformerFusion(16, layers=1, heads=4).train()
    mask = torch.tensor([[True] * 6, [False, True, False, False, False, False]])
    result = fusion(torch.randn(2, 6, 16), mask, camera_dropout=1.0)
    assert result["mask"].any(1).all()


def test_crop_batching_preserves_embeddings_and_gradients():
    model = SpatialEncoder(TinyBackbone(), 16, heads=4, dropout=0).eval()
    images = torch.randn(1, 6, 4, 3, 8, 8)
    mask = torch.tensor([[True, False, True, True, False, True]])
    first = model(images, mask).camera_tokens
    first.square().sum().backward()
    gradient = model.backbone.projection.weight.grad.clone()
    model.zero_grad(set_to_none=True)
    model.backbone_batch_size = 4
    second = model(images, mask).camera_tokens
    second.square().sum().backward()
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(gradient, model.backbone.projection.weight.grad, atol=1e-5, rtol=1e-3)
