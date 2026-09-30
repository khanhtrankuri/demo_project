from torch import nn


class CountHead(nn.Module):
    """Per-class 0/1/2/3/4+ classification."""
    def __init__(self, width: int, classes: int, bins: int = 5):
        super().__init__()
        self.classes, self.bins = classes, bins
        self.projection = nn.Linear(width, classes * bins)

    def forward(self, tokens):
        return self.projection(tokens).unflatten(-1, (self.classes, self.bins))
