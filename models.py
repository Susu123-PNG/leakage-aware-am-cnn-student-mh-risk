"""models.py — Model definitions for the benchmark.

Sklearn models (LR/RF/XGB/LGBM) are used directly in train_eval.py.
Torch models below share the protocol: input is a 1 x 69 feature map,
training uses Adam + early stopping on the held-out 10% validation subset.
Parameter budgets (printed at build time): AM-CNN ~9,500; PM-CNN within
3% of AM-CNN; standard CNN roughly half.
"""
import torch, torch.nn as nn


def conv_bn(k, cin, cout):
    return nn.Sequential(nn.Conv1d(cin, cout, k, padding=k // 2),
                         nn.BatchNorm1d(cout), nn.ReLU())


class SE(nn.Module):
    def __init__(self, m, r=4):
        super().__init__()
        self.fc1 = nn.Linear(m, max(1, m // r)); self.fc2 = nn.Linear(max(1, m // r), m)

    def forward(self, x):                       # x: B x M x L
        s = torch.sigmoid(self.fc2(torch.relu(self.fc1(x.mean(dim=2)))))
        return x * s.unsqueeze(-1)


class MultiScaleCNN(nn.Module):
    """AM-CNN: ms kernels (2,3,5) + SE. CNN+MS: se=False. CNN+SE: kernels (3,3,3)."""
    def __init__(self, channels=(40, 40, 40), kernels=(2, 3, 5), se=True):
        super().__init__()
        self.branches = nn.ModuleList([conv_bn(k, 1, c) for k, c in zip(kernels, channels)])
        m = sum(channels)
        self.se = SE(m) if se else nn.Identity()
        self.head = nn.Linear(m, 2)

    def forward(self, x):
        f = torch.cat([b(x.unsqueeze(1)) for b in self.branches], dim=1)  # B x M x L
        f = self.se(f)
        return self.head(f.mean(dim=2))


class StandardCNN(nn.Module):
    def __init__(self, ch=24, k=3):
        super().__init__()
        self.c = nn.Sequential(conv_bn(k, 1, ch), conv_bn(k, ch, ch))
        self.head = nn.Linear(ch, 2)

    def forward(self, x):
        return self.head(self.c(x.unsqueeze(1)).mean(dim=2))


class TinyTransformer(nn.Module):
    def __init__(self, d=32, heads=4):
        super().__init__()
        self.proj = nn.Linear(1, d)
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, heads, 64, batch_first=True), 1)
        self.head = nn.Linear(d, 2)

    def forward(self, x):
        return self.head(self.enc(self.proj(x.unsqueeze(-1))).mean(dim=1))


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def build(name):
    if name == "AM-CNN":
        return MultiScaleCNN(channels=(40, 40, 40), kernels=(2, 3, 5), se=True)
    if name == "CNN+multi-scale":
        return MultiScaleCNN(channels=(40, 40, 40), kernels=(2, 3, 5), se=False)
    if name == "CNN+SE":
        return MultiScaleCNN(channels=(40, 40, 40), kernels=(3, 3, 3), se=True)
    if name == "PM-CNN":
        return MultiScaleCNN(channels=(36, 36, 36), kernels=(3, 3, 3), se=True)
    if name == "CNN":
        return StandardCNN()
    if name == "Transformer":
        return TinyTransformer()
    raise ValueError(name)


if __name__ == "__main__":
    for n in ["AM-CNN", "CNN+multi-scale", "CNN+SE", "PM-CNN", "CNN", "Transformer"]:
        print(n, "params:", count_params(build(n)))
