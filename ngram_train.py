import os
import random
import argparse

import torch
import torch.nn as nn

from ngram import NGramModel


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', required=True, type=str, help='Path to the dataset')
    parser.add_argument('--checkpoint_path', required=True, type=str, help='Path to save the model weights')
    parser.add_argument('--n_gram', type=int, default=4, help='N-gram size')
    parser.add_argument('--seed', type=int, default=random.randint(0, 999999), help='Random seed')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=10, help='Learning rate')
    args = parser.parse_args()

    names = open(args.dataset_path, 'r').read().splitlines()
    vocab = ['.'] + sorted(set(''.join(names)))
    vocab_size = len(vocab)

    # ordinal encoder/decoder
    itos = dict(enumerate(vocab))
    stoi = dict((s, i) for i, s in itos.items())

    # Creating X and Y dataset
    x_train = []
    y_train = []

    for name in names:
        name = '.' * (args.n_gram - 1) + '%s.' % name
        for chars in zip(name, *[name[i+1:] for i in range(args.n_gram-1)]):
            x_train.append([stoi[i] for i in chars[:-1]])
            y_train.append(stoi[chars[-1]])

    x_train = torch.tensor(x_train)
    y_train = torch.tensor(y_train)

    xenc = nn.functional.one_hot(
        x_train,
        num_classes=vocab_size
    ).view(x_train.shape[0], -1).float()

    g = torch.Generator().manual_seed(args.seed)
    model = NGramModel(
        vocab_size=vocab_size,
        n_gram=args.n_gram,
        generator=g
    )
    if os.path.exists(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])

    model.train()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        logits = model(xenc)
        loss = nn.functional.cross_entropy(logits, y_train)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{args.epochs}: Loss {loss}')

    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': model.vocab_size,
        'n_gram': model.n_gram,
        'itos': itos
    }, args.checkpoint_path)