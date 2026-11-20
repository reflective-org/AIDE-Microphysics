"""Network classes of the shipped tomas40 emulators. Same parameter names and math as in training,
so the published state_dicts load unchanged.

  BinTF   one token per bin (+2 gas tokens), all 12 half-hour outputs per token  (bintf_d384, bintf_d256)
  PackTF  4 bins per token (10 tokens + 1 gas token), 6 h output only            (cs_b)

Every model is wrapped to the same deployment signature
    step6h(s_norm (B, 82), c_norm (B, 13, 6)) -> (B, 82)  normalized log-state at +6 h.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_BINS = 40
T = 12                       # 30-min outputs per 6 h block (BinTF)
P = N_BINS + 2               # BinTF tokens per box


# ---------------------------------------------------------------- BinTF
def _enc(d, nl, nh=4):
    layer = nn.TransformerEncoderLayer(d, nh, 2 * d, dropout=0.0, batch_first=True,
                                       norm_first=True, activation="gelu")
    return nn.TransformerEncoder(layer, nl)


class BinTF(nn.Module):
    """42 tokens; token = [Nk0, Mk0] + gases + all 12 forcing vectors (72) -> d; output 12 x 2 per token."""

    def __init__(self, d, nl):
        super().__init__()
        self.tok_in = nn.Linear(2 + 2 + 6 * T, d)
        self.bin_emb = nn.Parameter(torch.randn(P, d) * 0.02)
        self.enc = _enc(d, nl)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 2 * T)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, s0, cond_seq):
        """s0 (B, 82), cond_seq (B, 13, 6), both normalized -> (B, 12, 82) normalized states at +0.5 .. +6 h."""
        B = s0.shape[0]
        nk, mk = s0[:, :N_BINS], s0[:, N_BINS:2 * N_BINS]
        gas = s0[:, 2 * N_BINS:]
        tok = torch.cat([torch.stack([nk, mk], -1), torch.stack([gas, torch.zeros_like(gas)], -1)], 1)
        glob = torch.cat([gas, cond_seq[:, 1:1 + T].flatten(1)], 1)
        x = self.tok_in(torch.cat([tok, glob.unsqueeze(1).expand(B, P, -1)], -1)) + self.bin_emb
        d = self.head(self.norm(self.enc(x))).reshape(B, P, T, 2).permute(0, 2, 1, 3)
        base = torch.cat([nk, mk, gas], -1).unsqueeze(1)
        return base + torch.cat([d[:, :, :N_BINS, 0], d[:, :, :N_BINS, 1], d[:, :, N_BINS:, 0]], -1)


# ---------------------------------------------------------------- PackTF (distilled students)
class Blk(nn.Module):
    def __init__(s, d, h, ff):
        super().__init__(); s.h = h; s.n1 = nn.LayerNorm(d); s.n2 = nn.LayerNorm(d)
        s.qkv = nn.Linear(d, 3 * d); s.o = nn.Linear(d, d)
        s.f1 = nn.Linear(d, ff); s.f2 = nn.Linear(ff, d)

    def forward(s, x):
        B, Tn, D = x.shape
        q = s.qkv(s.n1(x)).view(B, Tn, 3, s.h, D // s.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q[0], q[1], q[2]).transpose(1, 2).reshape(B, Tn, D)
        x = x + s.o(a)
        return x + s.f2(F.gelu(s.f1(s.n2(x))))


class PackTF(nn.Module):
    """Input: x (B, 82) normalized log-state, c16 (B, 16) = [logP, T, RH, 13 x logOH] normalized.
    Bins packed `pack` per token (40/pack tokens + 1 gas token), pre-norm transformer, residual output."""

    def __init__(s, d=256, nl=4, pack=4, heads=4, ff_mult=2, n_cond=16):
        super().__init__(); s.pack = pack; s.nt = N_BINS // pack; s.n_cond = n_cond
        s.tin = nn.Linear(2 * pack + n_cond + 2, d); s.gin = nn.Linear(2 + n_cond + 2, d)
        s.pos = nn.Parameter(torch.randn(s.nt + 1, d) * 0.02)
        s.blocks = nn.ModuleList([Blk(d, heads, ff_mult * d) for _ in range(nl)]); s.norm = nn.LayerNorm(d)
        s.hb = nn.Linear(d, 2 * pack); s.hg = nn.Linear(d, 2)
        for lin in (s.hb, s.hg):
            nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)

    def forward(s, x, c16):
        B = x.shape[0]; p = s.pack; nk, mk, gas = x[:, :N_BINS], x[:, N_BINS:2 * N_BINS], x[:, 2 * N_BINS:]
        bt = torch.stack([nk.view(B, s.nt, p), mk.view(B, s.nt, p)], -1).reshape(B, s.nt, 2 * p)
        glob = torch.cat([gas, c16], -1)
        tok = torch.cat([bt, glob[:, None].expand(B, s.nt, -1)], -1)
        t = torch.cat([s.tin(tok), s.gin(torch.cat([gas, c16, gas], -1))[:, None]], 1) + s.pos
        for b in s.blocks:
            t = b(t)
        t = s.norm(t); db = s.hb(t[:, :s.nt]).view(B, s.nt, p, 2); dg = s.hg(t[:, s.nt])
        return x + torch.cat([db[..., 0].reshape(B, N_BINS), db[..., 1].reshape(B, N_BINS), dg], 1)


class Mlp6(nn.Module):
    """(B, 13, 6) forcing -> the students' 16 inputs: P, T, RH at t0 + log OH at all 13 points."""

    def __init__(s, net):
        super().__init__(); s.net = net

    def forward(s, x, c):
        return s.net(x, torch.cat([c[:, 0, :3], c[:, :, 3]], 1))


def build(arch, cfg):
    """(net, wrap, last): `net` loads the published state_dict; `wrap(net)` is the module that is compiled;
    `last` = take the final half-hour of a (B, 12, 82) output outside the compiled graph (BinTF)."""
    if arch == "bintf":
        net = BinTF(cfg["d"], cfg["nl"])
        return net, net, True
    if arch == "packtf":
        net = PackTF(cfg["d"], cfg["nl"], cfg["pack"], cfg.get("heads", 4), cfg.get("ffm", 2))
        return net, Mlp6(net), False
    raise ValueError(f"unknown arch {arch!r}")
