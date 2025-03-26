import torch
import torch.nn as nn


class NGramModel(nn.Module):
    def __init__(self, vocab_size, n_gram, generator=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_gram = n_gram
        self.linear = nn.Linear(
            vocab_size * (n_gram-1),
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