"""Streaming water segmentation models on range-compressed rows.

All models map an input sequence (B, T, C, W) -> logits (B, T, W), causally in T (slow time):
output step t only depends on rows <= t. With a fixed latency D, output t is trained to predict
row t - D, which gives the model the remaining half synthetic aperture as look-ahead.
For onboard use, `step()` processes one row at a time with an explicit state.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(cin, cout, k=5, dilation=1):
    return nn.Sequential(nn.Conv1d(cin, cout, k, padding=dilation * (k // 2), dilation=dilation),
                         nn.BatchNorm1d(cout), nn.ReLU(inplace=True))


class RowEncoder(nn.Module):
    """1D CNN along range applied independently to every row."""

    def __init__(self, cin, ch):
        super().__init__()
        self.net = nn.Sequential(conv_block(cin, ch), conv_block(ch, ch, dilation=2),
                                 conv_block(ch, ch, dilation=4))

    def forward(self, x):  # (N, C, W)
        return self.net(x)


class ConvGRUCell1d(nn.Module):
    def __init__(self, cin, ch, k=5):
        super().__init__()
        self.ch = ch
        self.gates = nn.Conv1d(cin + ch, 2 * ch, k, padding=k // 2)
        self.cand = nn.Conv1d(cin + ch, ch, k, padding=k // 2)

    def forward(self, x, h):
        z, r = torch.sigmoid(self.gates(torch.cat([x, h], 1))).chunk(2, 1)
        n = torch.tanh(self.cand(torch.cat([x, r * h], 1)))
        return (1 - z) * h + z * n


class ConvGRUNet(nn.Module):
    """Row encoder -> ConvGRU over slow time (state = feature map along range) -> 1x1 head."""

    def __init__(self, cin=3, ch=32, layers=2):
        super().__init__()
        self.enc = RowEncoder(cin, ch)
        self.cells = nn.ModuleList([ConvGRUCell1d(ch, ch) for _ in range(layers)])
        self.head = nn.Sequential(conv_block(2 * ch, ch, k=3), nn.Conv1d(ch, 1, 1))

    def init_state(self, b, w, device="cpu"):
        return [torch.zeros(b, c.ch, w, device=device) for c in self.cells]

    def step(self, x, state):
        """x: (B, C, W) one row; returns logits (B, W) and new state."""
        f = self.enc(x)
        h_in, new = f, []
        for cell, h in zip(self.cells, state):
            h_in = cell(h_in, h)
            new.append(h_in)
        return self.head(torch.cat([f, h_in], 1))[:, 0], new

    def forward(self, x, state=None):
        b, t, c, w = x.shape
        state = state or self.init_state(b, w, x.device)
        f_all = self.enc(x.reshape(b * t, c, w)).reshape(b, t, -1, w)  # encoder in parallel
        outs = []
        for i in range(t):
            h_in, new = f_all[:, i], []
            for cell, h in zip(self.cells, state):
                h_in = cell(h_in, h)
                new.append(h_in)
            state = new
            outs.append(self.head(torch.cat([f_all[:, i], h_in], 1))[:, 0])
        return torch.stack(outs, 1), state


class BinGRUNet(nn.Module):
    """Row encoder -> independent GRU per range bin over slow time -> conv head along range.

    After range compression, a target's energy is already focused in range; what remains is the
    azimuth (slow-time) smear over the synthetic aperture. So the recurrence only needs to run along
    slow time per range bin (shared weights), while range context comes from the convolutional
    encoder and head. This maps to PyTorch's fused GRU kernel and is much cheaper than a ConvGRU."""

    def __init__(self, cin=3, ch=32, layers=1):
        super().__init__()
        self.ch, self.layers = ch, layers
        self.enc = RowEncoder(cin, ch)
        self.gru = nn.GRU(ch, ch, num_layers=layers, batch_first=True)
        self.head = nn.Sequential(conv_block(2 * ch, ch, k=5), nn.Conv1d(ch, 1, 1))

    def init_state(self, b, w, device="cpu"):
        return torch.zeros(self.layers, b * w, self.ch, device=device)

    def forward(self, x, state=None):
        b, t, c, w = x.shape
        f = self.enc(x.reshape(b * t, c, w)).reshape(b, t, self.ch, w)
        seq = f.permute(0, 3, 1, 2).reshape(b * w, t, self.ch)          # (B*W, T, ch)
        h, state = self.gru(seq, state if state is not None else self.init_state(b, w, x.device))
        h = h.reshape(b, w, t, self.ch).permute(0, 2, 3, 1)               # (B, T, ch, W)
        z = torch.cat([f, h], 2).reshape(b * t, 2 * self.ch, w)
        return self.head(z).reshape(b, t, w), state

    def step(self, x, state):
        """x: (B, C, W) one row; returns logits (B, W) and new state."""
        out, state = self.forward(x[:, None], state)
        return out[:, 0], state


class CausalTCN(nn.Module):
    """Baseline: row encoder -> causal dilated 2D conv over (slow time, range)."""

    def __init__(self, cin=3, ch=32, dilations=(1, 2, 4, 8, 16, 32)):
        super().__init__()
        self.enc = RowEncoder(cin, ch)
        self.dil = dilations
        self.convs = nn.ModuleList([nn.Conv2d(ch, ch, (2, 3), dilation=(d, 1), padding=(0, 1))
                                    for d in dilations])
        self.norms = nn.ModuleList([nn.BatchNorm2d(ch) for _ in dilations])
        self.head = nn.Conv2d(ch, 1, 1)

    @property
    def receptive_field(self):
        return 1 + sum(self.dil)

    def forward(self, x, state=None):
        b, t, c, w = x.shape
        f = self.enc(x.reshape(b * t, c, w)).reshape(b, t, -1, w).permute(0, 2, 1, 3)  # (B,ch,T,W)
        for conv, bn, d in zip(self.convs, self.norms, self.dil):
            f = f + F.relu(bn(conv(F.pad(f, (0, 0, d, 0)))))
        return self.head(f)[:, 0], None


def build_model(name: str, **kw) -> nn.Module:
    return {"convgru": ConvGRUNet, "bingru": BinGRUNet, "tcn": CausalTCN}[name](**kw)
