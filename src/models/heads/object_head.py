from torch import nn


class ObjectHead(nn.Module):
    def __init__(self, width: int, classes: int):
        super().__init__()
        self.projection = nn.Linear(width, classes)

    def forward(self, tokens):
        return self.projection(tokens)
