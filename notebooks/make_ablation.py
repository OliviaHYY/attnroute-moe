# ══════════════════════════════════════════════════════════════════════════════
# E10 (Version A) and E10b (Random Prior) ablations
#
# PURPOSE:
#   E10  — Version A: prior only, no W_r. Designed lower bound.
#           Tests whether the attention prior alone is sufficient.
#   E10b — Random prior: fixed random per-position signal + W_r.
#           Tests whether ANY structured signal helps, or semantic
#           attention structure specifically is what reduces collapse.
#
# BOTH run on Tiny ImageNet, 45 epochs, same setup as E4 Version B T50%.
# This makes all three directly comparable in the paper.
#
# FIGURES PRODUCED:
#   e10_training_curves.png
#   e10b_training_curves.png
#   ablation_comparison.png   ←  the key figure: V-MoE / Random / VB / VA
#   ablation_table.txt  ←  Latex for Table 3
# ══════════════════════════════════════════════════════════════════════════════

import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from transformers import ViTModel, ViTConfig


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Shared building blocks (self-contained, no imports from week5)
# ─────────────────────────────────────────────────────────────────────────────

class ExpertFFN(nn.Module):
    def __init__(self, d=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d)
        )
    def forward(self, x): return self.net(x)


class AttentionPrior(nn.Module):
    """Semantic prior from attention: fᵢ = [−H, Var, CLS-sim] → Pᵢ ∈ ℝᴱ"""
    def __init__(self, E=4, eps=1e-8):
        super().__init__()
        self.E = E; self.eps = eps
        init = torch.zeros(E, 3)
        for j in range(min(E, 3)): init[j, j] = 1.0
        if E > 3: init[3] = torch.tensor([1/3, 1/3, 1/3])
        self.prototypes = nn.Parameter(init)
        self.register_buffer('proto_init', init.clone())

    def forward(self, A):
        Ap  = A[:, :, 1:, :]
        Hi  = -(Ap * (Ap + self.eps).log()).sum(-1).mean(1)
        Vi  = ((Ap - Ap.mean(1, keepdim=True))**2).mean(-1).mean(1)
        Ac  = F.normalize(A[:, :, 0, :].mean(1), dim=-1)
        Ap2 = F.normalize(Ap.mean(1), dim=-1)
        Ci  = (Ap2 * Ac.unsqueeze(1)).sum(-1)
        fi  = torch.stack([-Hi, Vi, Ci], dim=-1)
        mu  = fi.mean(dim=[0, 1], keepdim=True)
        std = fi.std(dim=[0, 1], keepdim=True) + self.eps
        return (fi - mu) / std @ self.prototypes.T

    def drift(self):
        return (self.prototypes - self.proto_init).norm().item()


class CosineAnnealSchedule:
    def __init__(self, total_steps, frac=0.5):
        self.T = int(total_steps * frac)
    def get_lambda(self, step):
        if step >= self.T: return 0.0
        return math.cos(math.pi * step / (2 * self.T)) ** 2


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — E10: Version A (prior only, no W_r)
# ─────────────────────────────────────────────────────────────────────────────

class VersionABlock(nn.Module):
    """
    Version A ablation: Sᵢ = Pᵢ only. No W_r.
    λ is fixed at 1.0 throughout — there is no annealing because there
    is nothing to anneal toward (no W_r to hand off to).

    This is a DESIGNED LOWER BOUND, not a proposed method.
    It tests whether the attention prior alone, without any task-specific
    learned router, is sufficient to route tokens sensibly.

    Expected outcomes:
        POSITIVE finding (within 1-2% of V-MoE accuracy):
            Prior alone captures enough routing signal.
            Version B's W_r adds marginal task-specific adaptation.
        EXPECTED outcome (3-5% below V-MoE accuracy):
            W_r contributes task-specific routing that the prior cannot supply.
            This justifies Version B's hybrid design.
        NEGATIVE finding (>5% below V-MoE):
            Prior is actively misdirecting tokens. Check:
            1. proto init (identity_like is correct)
            2. batch norm in AttentionPrior.forward()
            3. Whether DeiT backbone is frozen
    """
    def __init__(self, d=768, E=4, k=2, lb=0.01):
        super().__init__()
        self.E = E; self.k = k; self.lb_coeff = lb
        self.prior   = AttentionPrior(E)
        self.experts = nn.ModuleList([ExpertFFN(d) for _ in range(E)])
        self.norm    = nn.LayerNorm(d)
        # NO self.W_r — that is the whole point

    def forward(self, z, A, lam=1.0):   # lam ignored
        zp = self.norm(z[:, 1:, :])
        Si = self.prior(A)               # [B, 196, E]
        tv, ti = torch.topk(Si, self.k, dim=-1)
        tw = torch.softmax(tv, dim=-1)
        out = torch.zeros_like(zp)
        for e in range(self.E):
            m = (ti == e)
            w = (tw * m.float()).sum(-1, keepdim=True)
            out += w * self.experts[e](zp)
        ap = torch.softmax(Si, dim=-1).mean(dim=[0, 1])
        lb_loss = self.lb_coeff * self.E * (ap * ap).sum()
        o = z.clone(); o[:, 1:, :] = o[:, 1:, :] + out
        return o, lb_loss

    @torch.no_grad()
    def routing_metrics(self, z, A, lam=1.0):
        zp = self.norm(z[:, 1:, :])
        Si = self.prior(A)
        _, ti = torch.topk(Si, self.k, dim=-1)
        c  = torch.tensor([(ti == e).float().sum().item() for e in range(self.E)])
        cv = (c.std() / (c.mean() + 1e-8)).item()
        p  = c / c.sum()
        ue = -(p * (p + 1e-8).log()).sum().item() / math.log(self.E)
        return {'cv': cv, 'util_entropy': ue,
                'expert_counts': c.int().tolist()}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — E10b: Random Prior block
# ─────────────────────────────────────────────────────────────────────────────

class RandomPriorBlock(nn.Module):
    """
    Random prior ablation: Sᵢ = W_r(zᵢ) + λ(t) · P_random

    P_random is a fixed random vector per token POSITION (not per token value).
    Shape: [196, E] — one fixed routing preference per spatial patch position.
    It is sampled ONCE at init and NEVER updated (register_buffer, not Parameter).

    This is architecturally identical to Version B EXCEPT:
        Version B:  P_random comes from DeiT attention maps (semantic)
        E10b:       P_random is fixed noise (no semantic content)

    The annealing schedule is IDENTICAL to Version B T50%:
        λ(0)   = 1.0  (random prior fully active)
        λ(ep23)= 0.0  (random prior silent, W_r takes over)

    The only difference between Version B (E4) and E10b is the CONTENT of Pᵢ.

    What to look for:
        If E10b CV@ep1 ≈ E4 (Version B) CV@ep1  (< 5% difference):
            → Any structured signal reduces collapse, not semantic content
            → Soften paper claim to "structured diversity helps"
            → Still publishable and scientifically interesting

        If E10b CV@ep1 is significantly higher than E4 (≥ 10% difference):
            → Semantic structure in DeiT attention specifically matters
            → Strong claim: "attention prior reduces collapse via semantic content"
            → This is the preferred outcome given H1's strong DeiT signal

        If E10b CV@ep1 < E4 CV@ep1 (random beats semantic):
            → Something wrong — check E4 implementation
            → This should not happen given H1 results

    Normalization note:
        P_random is L2-normalized per token position so its scale matches
        a typical semantic prior output. Without normalization, P_random might
        dominate W_r logits through scale rather than structure.
    """
    def __init__(self, d=768, E=4, k=2, lb=0.01, seed=42):
        super().__init__()
        self.E = E; self.k = k; self.lb_coeff = lb

        self.W_r     = nn.Linear(d, E, bias=False)
        self.experts = nn.ModuleList([ExpertFFN(d) for _ in range(E)])
        self.norm    = nn.LayerNorm(d)

        # Fixed random prior: [196, E], sampled once, never updated
        # Seed is fixed so results are reproducible across runs
        g = torch.Generator(); g.manual_seed(seed)
        rand_p = torch.randn(196, E, generator=g)  # [196, E]
        # L2-normalize each row so scale is consistent with semantic prior
        rand_p = rand_p / (rand_p.norm(dim=-1, keepdim=True) + 1e-8)
        self.register_buffer('rand_prior', rand_p)  # frozen, not a Parameter

    def forward(self, z, A, lam=0.0):
        """
        Args:
            z:   [B, 197, 768]
            A:   [B, 12, 197, 197]  — ignored (random prior doesn't use A)
            lam: current λ(t) from schedule
        """
        zp = self.norm(z[:, 1:, :])                      # [B, 196, d]
        Pi = self.rand_prior.unsqueeze(0)                 # [1, 196, E] → broadcast
        Si = self.W_r(zp) + lam * Pi                     # [B, 196, E]

        tv, ti = torch.topk(Si, self.k, dim=-1)
        tw = torch.softmax(tv, dim=-1)
        out = torch.zeros_like(zp)
        for e in range(self.E):
            m = (ti == e)
            w = (tw * m.float()).sum(-1, keepdim=True)
            out += w * self.experts[e](zp)

        ap = torch.softmax(Si, dim=-1).mean(dim=[0, 1])
        lb_loss = self.lb_coeff * self.E * (ap * ap).sum()
        o = z.clone(); o[:, 1:, :] = o[:, 1:, :] + out
        return o, lb_loss

    @torch.no_grad()
    def routing_metrics(self, z, A, lam=0.0):
        zp = self.norm(z[:, 1:, :])
        Pi = self.rand_prior.unsqueeze(0)
        Si = self.W_r(zp) + lam * Pi
        _, ti = torch.topk(Si, self.k, dim=-1)
        c  = torch.tensor([(ti == e).float().sum().item() for e in range(self.E)])
        cv = (c.std() / (c.mean() + 1e-8)).item()
        p  = c / c.sum()
        ue = -(p * (p + 1e-8).log()).sum().item() / math.log(self.E)
        return {'cv': cv, 'util_entropy': ue,
                'expert_counts': c.int().tolist()}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Wrapper and training infrastructure
# ─────────────────────────────────────────────────────────────────────────────

class WrappedViT(nn.Module):
    def __init__(self, backbone, moe_block, num_classes=200):
        super().__init__()
        self.backbone  = backbone
        self.moe_block = moe_block
        self.head      = nn.Linear(768, num_classes)
        for p in backbone.parameters():
            p.requires_grad = False

    def forward(self, pv, lam=0.0):
        out    = self.backbone(pixel_values=pv, output_attentions=True)
        z      = out.last_hidden_state
        A      = out.attentions[-1]
        zo, lb = self.moe_block(z, A, lam=lam)
        return self.head(zo[:, 0, :]), lb


class ExperimentLogger:
    def __init__(self, name):
        self.name = name; self.history = []; self.t0 = time.time()

    def log(self, epoch, train_m, val_m, lam, drift=None):
        entry = {
            'epoch': epoch, 'lambda': lam,
            'train_loss': train_m['loss'], 'train_acc': train_m['acc'],
            'val_acc': val_m['val_acc'], 'cv': val_m['cv'],
            'util_entropy': val_m['util_entropy'],
            'expert_counts': val_m['expert_counts'],
            'drift': drift,
            'elapsed_min': (time.time() - self.t0) / 60,
        }
        self.history.append(entry)
        d_str = f" | drift={drift:.4f}" if drift is not None else ""
        print(f"  Ep {epoch:3d} | λ={lam:.3f} | "
              f"loss={train_m['loss']:.4f} | "
              f"acc={val_m['val_acc']*100:.2f}% | "
              f"CV={val_m['cv']:.4f} | "
              f"H={val_m['util_entropy']:.3f}{d_str}")

    def steps_to_pct(self, pct=0.90):
        if not self.history: return None
        target = max(e['val_acc'] for e in self.history) * pct
        for e in self.history:
            if e['val_acc'] >= target: return e['epoch']

    def summary(self):
        if not self.history: return
        f = self.history[-1]; ep1 = self.history[0]
        print(f"\n{'='*55}")
        print(f"  {self.name}")
        print(f"  CV @ epoch 1:  {ep1['cv']:.4f}")
        print(f"  CV @ epoch 5:  {self.history[4]['cv']:.4f}" if len(self.history)>4 else "")
        print(f"  Final acc:     {f['val_acc']*100:.2f}%")
        print(f"  Steps to 90%: ep{self.steps_to_pct()}")
        if f['drift'] is not None:
            print(f"  Final drift:   {f['drift']:.4f}")
        print(f"{'='*55}")

    def save(self, path):
        with open(path, 'w') as fh:
            json.dump({'name': self.name, 'history': self.history}, fh, indent=2)
        print(f"  Log → {path}")

    @classmethod
    def load(cls, path):
        with open(path) as fh: d = json.load(fh)
        obj = cls(d['name']); obj.history = d['history']
        return obj


def train_one_epoch(model, loader, optimizer, schedule, step, device):
    model.train()
    total_loss = total_correct = total_n = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        lam = schedule.get_lambda(step)
        logits, lb = model(imgs, lam=lam)
        loss = F.cross_entropy(logits, labels) + lb
        optimizer.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss    += loss.item() * imgs.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_n       += imgs.size(0)
        step          += 1
    return {'loss': total_loss/total_n, 'acc': total_correct/total_n, 'step': step}


@torch.no_grad()
def validate(model, loader, moe_block, schedule, step, device):
    model.eval()
    total_correct = total_n = 0; all_metrics = []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        lam = schedule.get_lambda(step)
        logits, _ = model(imgs, lam=lam)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_n       += imgs.size(0)
        out = model.backbone(pixel_values=imgs, output_attentions=True)
        z = out.last_hidden_state; A = out.attentions[-1]
        all_metrics.append(moe_block.routing_metrics(z, A, lam))
    return {
        'val_acc':      total_correct / total_n,
        'cv':           np.mean([m['cv']           for m in all_metrics]),
        'util_entropy': np.mean([m['util_entropy'] for m in all_metrics]),
        'expert_counts':np.sum( [m['expert_counts'] for m in all_metrics],
                                axis=0).tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Data loaders
# ─────────────────────────────────────────────────────────────────────────────

def get_tinyimagenet_loaders(data_dir, batch_size=256, num_workers=4):
    import shutil
    mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.Resize(256), transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(), transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(mean, std),
    ])
    train_dir = os.path.join(data_dir, 'train')
    val_dir   = os.path.join(data_dir, 'val')
    val_org   = os.path.join(data_dir, 'val_organized')
    if not os.path.exists(val_org):
        print("  Reorganizing val/ ...")
        ann = os.path.join(val_dir, 'val_annotations.txt')
        with open(ann) as f:
            for line in f:
                parts = line.strip().split('\t')
                src = os.path.join(val_dir, 'images', parts[0])
                dst = os.path.join(val_org, parts[1])
                os.makedirs(dst, exist_ok=True)
                shutil.copy(src, os.path.join(dst, parts[0]))
        print(f"  Done → {val_org}")
    train_ds = datasets.ImageFolder(train_dir, train_tf)
    val_ds   = datasets.ImageFolder(val_org,   val_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                               shuffle=True,  num_workers=num_workers,
                               pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                               shuffle=False, num_workers=num_workers,
                               pin_memory=True)
    print(f"  {len(train_ds):,} train | {len(val_ds):,} val | "
          f"{len(train_loader)} steps/epoch")
    return train_loader, val_loader


def load_deit(device='cpu'):
    print("  Loading facebook/deit-base-patch16-224 ...")
    model = ViTModel.from_pretrained(
        "facebook/deit-base-patch16-224",
        output_attentions=True, attn_implementation="eager",
    )
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    return model.to(device)


def load_backbone_smoke(device='cpu'):
    config = ViTConfig(
        hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
        intermediate_size=3072, image_size=224, patch_size=16,
        num_channels=3, attn_implementation="eager",
    )
    model = ViTModel(config).eval()
    for p in model.parameters(): p.requires_grad = False
    return model.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Single experiment runner
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(exp_id, backbone, train_loader, val_loader,
                   epochs=45, device='cpu', out_dir='.', num_classes=200):
    """
    exp_id: 'E10' or 'E10b'
    Both use T_anneal=50% and 45 epochs to match E4 (Version B T50%).
    """
    print(f"\n{'─'*55}")
    print(f"  Running {exp_id}")
    print(f"{'─'*55}")

    total_steps = epochs * len(train_loader)

    if exp_id == 'E10':
        moe_block = VersionABlock(E=4, k=2, lb=0.01).to(device)
        run_name  = 'E10_VersionA_TinyIN_E4k2'
        # Version A uses λ=1.0 fixed — no annealing, but we still pass
        # a schedule object for API consistency. T=infinity means λ never hits 0.
        schedule  = CosineAnnealSchedule(total_steps, frac=999.0)
        # ^ frac=999 → T=999*steps → λ stays ≈1.0 the whole time

    elif exp_id == 'E10b':
        moe_block = RandomPriorBlock(E=4, k=2, lb=0.01, seed=42).to(device)
        run_name  = 'E10b_RandomPrior_TinyIN_E4k2_T50'
        # Same annealing as Version B T50% — fair comparison
        schedule  = CosineAnnealSchedule(total_steps, frac=0.50)
    else:
        raise ValueError(f"Unknown exp_id: {exp_id}")

    model = WrappedViT(backbone, moe_block, num_classes).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=0.05)
    lr_sched  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-6
    )

    logger    = ExperimentLogger(run_name)
    step      = 0
    best_acc  = 0.0
    ckpt_path = os.path.join(out_dir, f"{run_name}_best.pt")

    for epoch in range(1, epochs + 1):
        train_m = train_one_epoch(model, train_loader, optimizer,
                                   schedule, step, device)
        step = train_m['step']
        lr_sched.step()

        val_m = validate(model, val_loader, moe_block, schedule, step, device)
        lam   = schedule.get_lambda(step)

        # drift only meaningful for E10 (Version A has prototypes that can drift)
        drift = None
        if exp_id == 'E10' and hasattr(moe_block, 'prior'):
            drift = moe_block.prior.drift()

        logger.log(epoch, train_m, val_m, lam, drift)

        if val_m['val_acc'] > best_acc:
            best_acc = val_m['val_acc']
            torch.save({
                'epoch':   epoch,
                'val_acc': best_acc,
                'state':   model.state_dict(),
            }, ckpt_path)

        if epoch % 10 == 0:
            print(f"  [keepalive] ep{epoch}/{epochs} | "
                  f"best={best_acc*100:.2f}% | "
                  f"{logger.history[-1]['elapsed_min']:.0f} min elapsed")

    # ── save and plot ──────────────────────────────────────────────────────────
    log_path = os.path.join(out_dir, f"{run_name}_log.json")
    logger.save(log_path)
    logger.summary()
    _plot_single(logger, out_dir)
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _plot_single(logger, out_dir):
    h    = logger.history
    eps  = [e['epoch']       for e in h]
    accs = [e['val_acc']*100 for e in h]
    cvs  = [e['cv']          for e in h]
    lams = [e['lambda']      for e in h]
    ues  = [e['util_entropy'] for e in h]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle(logger.name, fontsize=11, fontweight='bold')

    axes[0,0].plot(eps, accs, 'b-', lw=2)
    axes[0,0].axhline(max(accs)*0.9, color='gray', ls='--', lw=1, label='90% of best')
    axes[0,0].set_xlabel('Epoch'); axes[0,0].set_ylabel('Val acc (%)')
    axes[0,0].set_title('(a) Validation accuracy'); axes[0,0].legend(fontsize=8)
    axes[0,0].grid(alpha=0.3)

    axes[0,1].plot(eps, cvs, 'r-', lw=2)
    if cvs:
        axes[0,1].annotate(f"CV@ep1={cvs[0]:.4f}",
                            xy=(eps[0], cvs[0]),
                            xytext=(eps[0]+2, cvs[0]+0.01),
                            fontsize=8, color='red')
    axes[0,1].set_xlabel('Epoch'); axes[0,1].set_ylabel('CV (collapse rate)')
    axes[0,1].set_title('(b) Expert collapse rate  ← lower is better')
    axes[0,1].grid(alpha=0.3)

    axes[1,0].plot(eps, lams, 'g-', lw=2)
    axes[1,0].fill_between(eps, lams, alpha=0.15, color='green')
    axes[1,0].set_xlabel('Epoch'); axes[1,0].set_ylabel('λ(t)')
    axes[1,0].set_title('(c) Annealing schedule')
    axes[1,0].grid(alpha=0.3)

    axes[1,1].plot(eps, ues, 'm-', lw=2)
    axes[1,1].set_ylim(0, 1.05)
    axes[1,1].set_xlabel('Epoch'); axes[1,1].set_ylabel('Util entropy (0–1)')
    axes[1,1].set_title('(d) Expert utilization')
    axes[1,1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, f"{logger.name}_curves.png")
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Curve → {path}")


def make_ablation_comparison(logs, out_dir):
    """
    The key figure: CV@ep1 and CV trajectory for
        E2  V-MoE           (red)
        E10b Random Prior   (orange)
        E4  Version B       (green)
        E10 Version A       (purple)

    What this figure should show:
        - Red line starts highest (most collapsed)
        - Orange line starts lower than red (any prior helps a little)
        - Green line starts lower than orange (semantic prior helps more)
        - Purple line may start very low (prior alone can route from step 1)
          OR start high if prior alone causes collapse
    """
    colors = {
        'E2_VMoE':      '#e41a1c',   # red
        'E10b_Random':  '#ff7f00',   # orange
        'E4_VersionB':  '#4daf4a',   # green
        'E10_VersionA': '#984ea3',   # purple
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Ablation comparison — Tiny ImageNet\n"
                 "Random Prior tests whether semantic attention structure "
                 "specifically matters", fontsize=11)

    for key, logger in logs.items():
        if logger is None: continue
        h    = logger.history
        eps  = [e['epoch']       for e in h]
        cvs  = [e['cv']          for e in h]
        accs = [e['val_acc']*100 for e in h]
        c    = colors.get(key, 'gray')
        name = key.replace('_', ' ')
        axes[0].plot(eps, cvs,  color=c, lw=2, label=name)
        axes[1].plot(eps, accs, color=c, lw=2, label=name)

    # annotate CV@ep1 for each method on the CV plot
    for key, logger in logs.items():
        if logger is None or not logger.history: continue
        cv1 = logger.history[0]['cv']
        axes[0].scatter([1], [cv1], color=colors.get(key,'gray'), s=60, zorder=5)

    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Expert collapse CV')
    axes[0].set_title('Expert collapse rate\n'
                       'Lower CV@ep1 = less initial collapse')
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Val accuracy (%)')
    axes[1].set_title('Validation accuracy')
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, 'week7_ablation_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"\n  Comparison figure → {path}")
    return path


def make_ablation_table(logs, out_dir):
    """
    Produces LaTeX for Table 4 of the paper.
    Compares V-MoE / Random Prior / Version A / Version B on:
        CV@ep1, CV@ep5, final accuracy, steps-to-90%
    """
    order = [
        ('E2_VMoE',      r'V-MoE~\citep{riquelme2021vmoe}',       'baseline'),
        ('E10b_Random',  r'Random prior $+$ $W_r$ (ablation)',      'ablation'),
        ('E10_VersionA', r'Version A: prior only (ablation)',        'ablation'),
        ('E4_VersionB',  r'\textbf{AttnRoute-MoE B (ours)}',       'proposed'),
    ]

    print(f"\n{'═'*70}")
    print("TABLE 4 — Ablation: V-MoE / Random Prior / Version A / Version B")
    print(f"{'Method':<35} {'Acc%':>6} {'CV@1':>7} {'CV@5':>7} "
          f"{'CVfin':>7} {'Steps90':>8}")
    print('─'*70)

    latex_rows = []
    for key, label, role in order:
        logger = logs.get(key)
        if logger is None or not logger.history:
            print(f"  {key}: log not found — skip")
            continue
        h    = logger.history
        f    = h[-1]
        ep1  = h[0]
        ep5  = h[4]['cv'] if len(h) > 4 else None
        s90  = logger.steps_to_pct(0.90)
        acc  = max(e['val_acc'] for e in h) * 100

        ep5s = f"{ep5:.4f}" if ep5 else "---"
        print(f"  {label.replace(chr(92),'').replace('{','').replace('}',''):<33} "
              f"{acc:>6.2f} {ep1['cv']:>7.4f} {ep5s:>7} "
              f"{f['cv']:>7.4f} {str(s90):>8}")

        latex_rows.append(
            f"{label} & {acc:.2f} & {ep1['cv']:.4f} & "
            f"{ep5s} & {f['cv']:.4f} & {s90} \\\\"
        )
    print('═'*70)

    # ── print interpretation ──────────────────────────────────────────────────
    vm  = logs.get('E2_VMoE')
    rnd = logs.get('E10b_Random')
    vb  = logs.get('E4_VersionB')
    va  = logs.get('E10_VersionA')

    if vm and rnd and vb:
        cv_vm  = vm.history[0]['cv']
        cv_rnd = rnd.history[0]['cv']
        cv_vb  = vb.history[0]['cv']

        rnd_imp = (cv_vm  - cv_rnd) / cv_vm  * 100
        vb_imp  = (cv_vm  - cv_vb)  / cv_vm  * 100
        margin  = (cv_rnd - cv_vb)  / cv_rnd * 100

        print(f"\n  INTERPRET:")
        print(f"    V-MoE CV@ep1:        {cv_vm:.4f}")
        print(f"    Random Prior CV@ep1: {cv_rnd:.4f}  ({rnd_imp:+.1f}% vs V-MoE)")
        print(f"    Version B CV@ep1:    {cv_vb:.4f}  ({vb_imp:+.1f}% vs V-MoE)")
        print(f"    VB advantage over random: {margin:+.1f}%")

        if margin >= 10:
            print(f"\n  ✓ SEMANTIC claim supported:")
            print(f"    Version B ({cv_vb:.4f}) clearly beats random prior "
                  f"({cv_rnd:.4f}).")
            print(f"    Paper claim: 'The benefit of AttnRoute-MoE comes from")
            print(f"    semantic attention structure, not merely structured diversity.'")
        elif margin >= 5:
            print(f"\n  ~ MARGINAL semantic advantage ({margin:.1f}%):")
            print(f"    Modest evidence for semantic content. Acknowledge in §6:")
            print(f"    'Version B outperforms random prior, suggesting semantic")
            print(f"    structure contributes, though the margin is modest.'")
        else:
            print(f"\n  ✗ NO semantic advantage ({margin:.1f}%):")
            print(f"    Random ≈ Semantic. Reframe §5.3 and §6:")
            print(f"    'The benefit of structured early routing priors comes from")
            print(f"    providing routing diversity, independent of semantic content.")
            print(f"    Any spatially structured signal during early training appears")
            print(f"    sufficient to prevent expert collapse.'")
            print(f"    This is still a valid and publishable finding.")

    # ── LaTeX table ───────────────────────────────────────────────────────────
    latex = '\n'.join([
        r'\begin{table}[h]',
        r'\centering',
        r'\small',
        r'\caption{%',
        r'  Ablation study: V-MoE baseline, random prior, Version A (prior only),',
        r'  and AttnRoute-MoE B. All on Tiny ImageNet with frozen DeiT-B/16,',
        r'  $E=4$, $k=2$, 45 epochs. The random prior ablation (row~2) uses a',
        r'  fixed per-position random vector instead of the attention prior,',
        r'  with the same annealing schedule as Version~B, isolating whether',
        r'  the benefit of AttnRoute-MoE comes from semantic attention structure',
        r'  or merely from injecting any structured signal during early training.',
        r'}',
        r'\label{tab:ablation}',
        r'\begin{tabular}{lccccr}',
        r'\toprule',
        r'Method & Top-1 (\%) & CV@ep1 $\downarrow$ & CV@ep5 & '
        r'CV@final & Steps-90\% \\',
        r'\midrule',
    ] + latex_rows + [r'\bottomrule', r'\end{tabular}', r'\end{table}'])

    tbl_path = os.path.join(out_dir, 'week7_ablation_table.txt')
    with open(tbl_path, 'w') as fh: fh.write(latex)
    print(f"\n  Table 4 LaTeX → {tbl_path}")
    return latex


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — Smoke test
# ─────────────────────────────────────────────────────────────────────────────

def smoke_test(out_dir='.'):
    print('='*60); print('WEEK 7 SMOKE TEST'); print('='*60)
    os.makedirs(out_dir, exist_ok=True)
    device = 'cpu'

    config = ViTConfig(
        hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
        intermediate_size=3072, image_size=224, patch_size=16,
        num_channels=3, attn_implementation="eager",
    )
    backbone = ViTModel(config).eval()
    for p in backbone.parameters(): p.requires_grad = False

    # quick unit tests before running anything
    print("\n[1/4] Unit tests ...")
    dummy_z = torch.randn(2, 197, 768)
    dummy_A = torch.softmax(torch.randn(2, 12, 197, 197), dim=-1)

    va = VersionABlock(E=4, k=2)
    o, lb = va(dummy_z, dummy_A, lam=1.0)
    assert o.shape == dummy_z.shape and not torch.isnan(o).any()
    m = va.routing_metrics(dummy_z, dummy_A)
    print(f"  VersionA: CV={m['cv']:.4f}  lb={lb.item():.4f}  ✓")

    rb = RandomPriorBlock(E=4, k=2, seed=42)
    o, lb = rb(dummy_z, dummy_A, lam=0.8)
    assert o.shape == dummy_z.shape and not torch.isnan(o).any()
    m = rb.routing_metrics(dummy_z, dummy_A, lam=0.8)
    print(f"  RandomPrior: CV={m['cv']:.4f}  lb={lb.item():.4f}  ✓")

    # confirm rand_prior is NOT a parameter (won't update)
    assert not any(p is rb.rand_prior for p in rb.parameters())
    print(f"  rand_prior is buffer (not Parameter) ✓")

    # confirm rand_prior is L2-normalized (rows have unit norm)
    norms = rb.rand_prior.norm(dim=-1)
    assert (norms - 1.0).abs().max() < 1e-5
    print(f"  rand_prior L2-normalized ✓")

    # annealing: E10 should have λ≈1 always; E10b should hit 0
    s_va  = CosineAnnealSchedule(100, frac=999.0)
    s_vb  = CosineAnnealSchedule(100, frac=0.50)
    assert s_va.get_lambda(99) > 0.99, "VersionA schedule broken"
    assert s_vb.get_lambda(99) == 0.0, "VersionB schedule broken"
    print(f"  Schedules correct ✓")

    # smoke train — 2 epochs on fake data
    print("\n[2/4] Smoke training E10 (2 epochs) ...")
    fake = torch.randn(16, 3, 224, 224)
    fake_l = torch.randint(0, 10, (16,))
    loader = DataLoader(TensorDataset(fake, fake_l), batch_size=4, shuffle=True)

    for exp_id in ['E10', 'E10b']:
        print(f"\n[3/4] {exp_id} ...")
        logger = run_experiment(exp_id, backbone, loader, loader,
                                 epochs=2, device=device,
                                 out_dir=out_dir, num_classes=10)
        assert logger.history[0]['cv'] >= 0
        print(f"  ✓ {exp_id} CV@ep1={logger.history[0]['cv']:.4f}")

    # fake comparison figure
    print("\n[4/4] Ablation comparison figure ...")
    logs = {
        'E2_VMoE':     None,
        'E10b_Random': ExperimentLogger.load(
            os.path.join(out_dir, 'E10b_RandomPrior_TinyIN_E4k2_T50_log.json')),
        'E10_VersionA':ExperimentLogger.load(
            os.path.join(out_dir, 'E10_VersionA_TinyIN_E4k2_log.json')),
        'E4_VersionB': None,
    }
    make_ablation_comparison(logs, out_dir)

    print(f"\n{'='*60}")
    print("SMOKE TEST COMPLETE")
    for fname in ['week7_ablation_comparison.png',
                  'E10_VersionA_TinyIN_E4k2_log.json',
                  'E10b_RandomPrior_TinyIN_E4k2_T50_log.json']:
        path = os.path.join(out_dir, fname)
        print(f"  {'✓' if os.path.exists(path) else '✗'} {fname}")
    print('='*60)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — Full run and compare mode
# ─────────────────────────────────────────────────────────────────────────────

def full_run(exp_id, data_dir, out_dir='/kaggle/working',
             epochs=45, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    os.makedirs(out_dir, exist_ok=True)

    backbone = load_deit(device)
    train_loader, val_loader = get_tinyimagenet_loaders(data_dir, batch_size=256)
    return run_experiment(exp_id, backbone, train_loader, val_loader,
                           epochs=epochs, device=device, out_dir=out_dir)


def compare_mode(log_dir, e2_log, e4_log, out_dir=None):
    """Load all four logs and produce Figure + Table."""
    if out_dir is None: out_dir = log_dir

    def try_load(filename):
        path = os.path.join(log_dir, filename)
        if os.path.exists(path):
            print(f"  Loaded: {filename}")
            return ExperimentLogger.load(path)
        print(f"  ✗ Not found: {filename}")
        return None

    logs = {
        'E2_VMoE':      try_load(e2_log),
        'E10b_Random':  try_load('E10b_RandomPrior_TinyIN_E4k2_T50_log.json'),
        'E10_VersionA': try_load('E10_VersionA_TinyIN_E4k2_log.json'),
        'E4_VersionB':  try_load(e4_log),
    }
    make_ablation_comparison(logs, out_dir)
    make_ablation_table(logs, out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',    choices=['smoke','full','compare'],
                        default='smoke')
    parser.add_argument('--exp',     default='E10b', help='E10 or E10b')
    parser.add_argument('--out_dir', default='.')
    parser.add_argument('--data_dir',default='/tmp')
    parser.add_argument('--epochs',  type=int, default=45)
    parser.add_argument('--e2_log',  default='E2_VMoE_TinyImageNet_E4k2_log.json')
    parser.add_argument('--e4_log',  default='E4_VersionB_TinyImageNet_E4k2_T50_log.json')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.mode == 'smoke':
        smoke_test(out_dir=args.out_dir)
    elif args.mode == 'full':
        full_run(args.exp, args.data_dir, args.out_dir, args.epochs)
    else:
        compare_mode(args.out_dir, args.e2_log, args.e4_log)
