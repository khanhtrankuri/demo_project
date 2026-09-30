"""Common contract used by SurroundSearch-v4."""
from __future__ import annotations

from abc import ABC, abstractmethod

from torch import nn


class VisionLanguageBackbone(nn.Module, ABC):
    projection_dim: int

    @abstractmethod
    def encode_image(self, pixel_values):
        raise NotImplementedError

    @abstractmethod
    def encode_text(self, tokens):
        raise NotImplementedError

    @abstractmethod
    def get_patch_tokens(self, pixel_values):
        raise NotImplementedError

    @abstractmethod
    def get_text_tokens(self, tokens):
        raise NotImplementedError
