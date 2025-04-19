import torch
import torch.nn as nn


class NGramModel(nn.Module):
    def __init__(self, vocab_size, batch_size, generator=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.linear = nn.Linear(
            vocab_size * batch_size,
            vocab_size,
            bias=False
        )
        torch.nn.init.normal_(
            self.linear.weight,
            -1, 1,
            generator=generator
        )

    def forward(self, X):
        return self.linear(X)