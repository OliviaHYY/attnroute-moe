# ═══════════════════════════════════════════════════════════════════
# TinyImageNet Experiments
# Table 2 + Figure 2
#
# EXPERIMENTS IN ORDER:
#   E1  Dense ViT-B baseline
#   E2  V-MoE (linear router)    ← baseline CV number to beat
#   E3  Expert Choice
#   E4  Version B (T_anneal=50%)  ← central claim
#
# PRODUCED:
#   e{N}_training_curves.png    per-experiment 4-panel training curves
#   cv_comparison.png    Figure 2: CV curves all methods overlaid
#   table2_tinyimagenet.txt  Table 2 TinyImageNet columns
# ════════════════════════════════════════════════════════════════════

import argparse
import json
import math
import os
import sys
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

# ──────────────────────────────────────────────────────
# SECTION 1 — Inline model definitions
# (self-contained so runs w/o import from other wks)
# ─────────────────────────────────────────────────────

class ExpertFFN(nn.Module):
  def __init__(self, d=768):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d)
    )
  def forward(self, x):
    return self.net(x)


class AttentionPrior(nn.Module):
  """
  Converts attention matrix A → prior routing distribution Pᵢ ∈ ℝᴱ.
  Signals: fᵢ = [−H(Aᵢ), Var_h(Aᵢ), cos(A_CLS, Aᵢ)]
  Prototypes: Pᵢ = fᵢ @ prototypes.T
  """
  def __init__(self, E=4, eps=1e-8):
    super().__init__()
    self.E   = E
    self.eps = eps
    # Identity-like init: expert j starts by preferring signal j
    init = torch.zeros(E, 3)
    for j in range(min(E, 3)):
        init[j, j] = 1.0
    if E > 3:
        init[3] = torch.tensor([1/3, 1/3, 1/3])
    self.prototypes = nn.Parameter(init)   # [E, 3]
    self.register_buffer('proto_init', init.clone()) # for drift track

  def forward(self, A, active=(0, 1, 2)):
    """A: [B, H, 197, 197] → Pᵢ: [B, 196, E]"""
    A_dtype = A.dtype
    A = A.float()

    Ap  = A[:, :, 1:, :]   # patch rows [B,H,196,197]
    # entropy
    Hi  = -(Ap * (Ap + self.eps).log()).sum(-1).mean(1)  # [B,196]

    # cross-head variance
    Am  = Ap.mean(1, keepdim=True)
    Vi  = ((Ap - Am)**2).mean(-1).mean(1)  # [B,196]

    # CLS cosine similarity
    Ac = F.normalize(A[:, :, 0, :], dim=-1, eps=1e-6)   # [B, H, 197]
    Ap_head = F.normalize(Ap, dim=-1, eps=1e-6)   # [B, H, 196, 197]
    Ci_per_head = (Ap_head * Ac.unsqueeze(2)).sum(-1)   # [B, H, 196]
    Ci_per_head = torch.clamp(Ci_per_head, min=-1.0, max=1.0)
    Ci = Ci_per_head.mean(1) # [B, 196]

    # assemble and batch-normalize
    fi  = torch.stack([-Hi, Vi, Ci], dim=-1)  # [B,196,3]
    # zero inactive signals BEFORE batch-norm so norm on active only
    mask = torch.zeros(3, device=fi.device)
    for s in active:
      mask[s] = 1.0
    fi = fi * mask
    mu  = fi.mean(dim=[0, 1], keepdim=True)
    std = fi.std( dim=[0, 1], keepdim=True) + 1e-6
    fi  = (fi - mu) / std

    out = fi @ self.prototypes.T.float()   # [B,196,E]
    return out.to(A_dtype)

  def drift(self):
    """Frobenius norm of prototype movement from init. (< 0.5)"""
    return (self.prototypes - self.proto_init).norm().item()


class AttnRouteMoEBlock(nn.Module):
  """Version B: Sᵢ = W_r(zᵢ) + λ(t)·Pᵢ (version A: no W_r)"""
  def __init__(self, d=768, E=4, k=2, version='B',
               lb=0.01, gamma=0.001, seed=42, active=(0, 1, 2)):
    super().__init__()
    self.E = E; self.k = k; self.version = version
    self.lb_coeff = lb; self.gamma = gamma
    self.active = active
    self.experts = nn.ModuleList([ExpertFFN(d) for _ in range(E)])
    self.norm = nn.LayerNorm(d)
    if version == 'B' or version == 'b':
      self.W_r = nn.Linear(d, E, bias=False)
    if version == 'b':
      g = torch.Generator()
      g.manual_seed(seed)
      rand_p = torch.randn(196, E, generator=g)
      wr_init_std = (1.0 / math.sqrt(E))
      rand_std = rand_p.std()
      rand_p = rand_p * (wr_init_std / rand_std)
      self.register_buffer('rand_prior', rand_p)
    else:
      self.prior = AttentionPrior(E)

  def forward(self, z, A, lam=0.0):
    zp = self.norm(z[:, 1:, :])   # [B,196,d]

    if self.version == 'b':
      Pi = self.rand_prior.unsqueeze(0)
    else:
      Pi = self.prior(A, active=self.active)    # [B,196,E]

    if self.version == 'B' or self.version == 'b':
        Si = self.W_r(zp) + lam * Pi
    else:
        Si = Pi  # Version A

    tv, ti = torch.topk(Si, self.k, dim=-1)
    tw = torch.softmax(tv, dim=-1)
    out = torch.zeros_like(zp)

    for e in range(self.E):
      m = (ti == e)
      w = (tw * m.float()).sum(-1, keepdim=True)
      out += w * self.experts[e](zp)
    # loss(es)
    ap = torch.softmax(Si, dim=-1).mean(dim=[0, 1])
    lb_loss = self.lb_coeff * self.E * (ap * ap).sum()
    # proto_loss = self.gamma * (self.prior.prototypes -
                              # self.prior.proto_init).pow(2).sum()
    o = z.clone()
    o[:, 1:, :] = o[:, 1:, :] + out
    # total_loss = lb_loss.unsqueeze(0) + proto_loss.unsqueeze(0)
    return o, lb_loss

  @torch.no_grad()
  def routing_metrics(self, z, A, lam=0.0):
    zp = self.norm(z[:, 1:, :])
    if self.version == 'b':
      Pi = self.rand_prior.unsqueeze(0)
    else:
      Pi = self.prior(A, active=self.active)
    if self.version == 'B' or self.version == 'b':
      Si = self.W_r(zp) + lam * Pi
    else:
      Si = Pi
    _, ti = torch.topk(Si, self.k, dim=-1)
    c  = torch.tensor([(ti == e).float().sum().item()
                                for e in range(self.E)])
    cv = (c.std() / (c.mean() + 1e-8)).item()
    p  = c / c.sum()
    ue = -(p * (p + 1e-8).log()).sum().item() / math.log(self.E)
    return {'cv': cv, 'util_entropy': ue,
            'expert_counts': c.int().tolist()}


class StandardMoEBlock(nn.Module):
  """V-MoE: Sᵢ = W_r(zᵢ) only. No prior."""
  def __init__(self, d=768, E=4, k=2, lb=0.01):
    super().__init__()
    self.E = E; self.k = k; self.lb_coeff = lb
    self.W_r = nn.Linear(d, E, bias=False)
    self.experts = nn.ModuleList([ExpertFFN(d) for _ in range(E)])
    self.norm = nn.LayerNorm(d)

  def forward(self, z, A=None, lam=None):
    zp = self.norm(z[:, 1:, :])  # [B, N, d]
    B, N, d = zp.shape

    Si = self.W_r(zp)  # [B, N, E]
    tv, ti = torch.topk(Si, self.k, dim=-1)  # [B, N, K]
    tw = torch.softmax(tv, dim=-1)  # [B, N, K]

    zp_flat  = zp.view(-1, d)  # [B*N, d]
    ti_flat  = ti.view(-1, self.k)  # [B*N, k]
    tw_flat  = tw.view(-1, self.k)  # [B*N, k]

    out_flat = torch.zeros_like(zp_flat)
    for e in range(self.E):
      mask = (ti_flat == e)
      if not mask.any():
        continue

      row_idx, col_idx = torch.where(mask)
      chosen_tokens = zp_flat[row_idx]  # [n_selected, d]
      expert_out = self.experts[e](chosen_tokens)
      routing_weights = tw_flat[row_idx, col_idx
                            ].unsqueeze(-1) # [n_selected, 1]
      out_flat.index_add_(0, row_idx, routing_weights * expert_out)

    out = out_flat.view(B, N, d)
    ap = torch.softmax(Si, dim=-1).mean(dim=[0, 1])
    lb_loss = self.lb_coeff * self.E * (ap * ap).sum()
    o = z.clone()
    o[:, 1:, :] = o[:, 1:, :] + out
    return o, lb_loss.unsqueeze(0)

  @torch.no_grad()
  def routing_metrics(self, z, A=None, lam=None):
    zp = self.norm(z[:, 1:, :])
    Si = self.W_r(zp)
    _, ti = torch.topk(Si, self.k, dim=-1)
    c  = torch.tensor([(ti == e).float().sum().item()
                              for e in range(self.E)])
    cv = (c.std() / (c.mean() + 1e-8)).item()
    p  = c / c.sum()
    ue = -(p * (p + 1e-8).log()).sum().item() / math.log(self.E)
    return {'cv': cv, 'util_entropy': ue,
            'expert_counts': c.int().tolist()}


class ExpertChoiceBlock(nn.Module):
  """Expert Choice: each expert selects top-c tokens.
     CV≈0 by construction."""
  def __init__(self, d=768, E=4, cf=2.0):
    super().__init__()
    self.E = E; self.cf = cf
    self.W_r     = nn.Linear(d, E, bias=False)
    self.experts = nn.ModuleList([ExpertFFN(d) for _ in range(E)])
    self.norm    = nn.LayerNorm(d)

  def forward(self, z, A=None, lam=None): # z:[B, N_tot, d]
    cls_token = z[:, :1, :] # Shape: [B, 1, d]
    patch_tokens = z[:, 1:, :] # Shape: [B, N, d]
    zp = self.norm(patch_tokens)   # [B, N, d]
    B, N, d = zp.shape
    c  = max(1, int(N * self.cf / self.E))

    S = zp @ self.W_r.weight.T   # [B, N, E]
    S_gated = torch.softmax(S, dim=-1)
    out = torch.zeros_like(zp)
    token_selection_counts = torch.zeros((B, N, 1),
                                device=zp.device, dtype=zp.dtype)
    for e in range(self.E):
      scores_for_expert = S_gated[:, :, e] # [B, N]
      vals, idx = torch.topk(S[:, :, e], c, dim=-1)  # [B,N] -> [B,c]
      sel = torch.gather(zp, 1,
            idx.unsqueeze(-1).expand(-1, -1, d)
        )  # [B, c, d]

      proc = self.experts[e](sel)  # [B, c, d]
      w = vals.unsqueeze(-1)  # Pre-normalized weight [B, c, 1]
      out.scatter_add_(1,
          idx.unsqueeze(-1).expand(-1, -1, d),
          w * proc.to(out.dtype))   # [B, c, d] -> [B, N, d]

      # Track selection frequency to balance downstream scales
      idx_3d = idx.unsqueeze(-1) # [B, c, 1]
      ones = torch.ones_like(idx_3d, dtype=token_selection_counts.dtype)
      token_selection_counts.scatter_add_(1, idx_3d, ones)

    scale_stabilizer = torch.clamp(token_selection_counts, min=1.0)
    out = out / scale_stabilizer

    modified_patches = patch_tokens + out
    o = torch.cat([cls_token, modified_patches], dim=1)
    return o, torch.tensor([0.0], device=z.device, requires_grad=True)

  @torch.no_grad()
  def routing_metrics(self, z, A=None, lam=None):
    zp = self.norm(z[:, 1:, :])
    B, N, _ = zp.shape
    c = max(1, int(N * self.cf / self.E))
    counts = torch.tensor([float(c * B)] * self.E)
    cv = (counts.std() / (counts.mean() + 1e-8)).item()
    p  = counts / counts.sum()
    ue = -(p * (p + 1e-8).log()).sum().item() / math.log(self.E)
    return {'cv': cv, 'util_entropy': ue,
            'expert_counts': counts.int().tolist()}


class WrappedViT(nn.Module):
  """Frozen backbone + MoE block + classification head."""
  def __init__(self, backbone, moe_block, num_classes=200):
    super().__init__()
    self.backbone = backbone
    self.moe_block = moe_block
    self.classifier = nn.Linear(backbone.config.hidden_size, num_classes)
    for p in backbone.parameters():
        p.requires_grad = False

  def forward(self, pv, lam=0.0):
    with torch.no_grad():
      out = self.backbone(pixel_values=pv, output_attentions=True)
      z = out.last_hidden_state
      A = out.attentions[-1]
    z_moe, lb_loss = self.moe_block(z, A=A, lam=lam)
    logits = self.classifier(z_moe[:, 0, :])

    if not self.training:
      # Fixed: Cleanly return logits and the pre-calculated metrics dictionary
      metrics = self.moe_block.routing_metrics(z.detach(), A.detach(), lam=lam)
      return logits, metrics

    return logits, lb_loss


class DenseViT(nn.Module):
  """E1 baseline: frozen backbone + FFN + head. No MoE."""
  def __init__(self, backbone, num_classes=200):
    super().__init__()
    self.backbone = backbone
    self.ffn = ExpertFFN(768)
    self.head = nn.Linear(768, num_classes)
    for p in backbone.parameters():
        p.requires_grad = False

  def forward(self, pv, lam=None):
    out = self.backbone(pixel_values=pv)
    z = out.last_hidden_state
    z = z + self.ffn(z)
    return self.head(z[:, 0, :]), torch.tensor(0.0, device=z.device)


class CosineAnnealSchedule:
  def __init__(self, total_steps, frac=0.5):
    self.T = int(total_steps * frac)

  def get_lambda(self, step):
    if step >= self.T: return 0.0
    return math.cos(math.pi * step / (2 * self.T)) ** 2


# ────────────────────────────
# SECTION 2 — Logger
# ────────────────────────────

class ExperimentLogger:
  def __init__(self, name):
    self.name    = name
    self.history = []
    self.t0      = time.time()

  def log(self, epoch, train_m, val_m, lam, drift=None):
    entry = {
        'epoch':        epoch,
        'lambda':       lam,
        'train_loss':   train_m['loss'],
        'train_acc':    train_m['acc'],
        'val_acc':      val_m['val_acc'],
        'cv':           val_m['cv'],
        'util_entropy': val_m['util_entropy'],
        'expert_counts':val_m['expert_counts'],
        'drift':        drift,
        'elapsed_min':  (time.time() - self.t0) / 60,
    }
    self.history.append(entry)
    drift_str = f" | drift={drift:.3f}" if drift is not None else ""
    print(f"  Ep {epoch:3d} | λ={lam:.3f} | "
          f"loss={train_m['loss']:.4f} | "
          f"acc={val_m['val_acc']*100:.1f}% | "
          f"CV={val_m['cv']:.4f} | "
          f"H={val_m['util_entropy']:.3f}{drift_str}")

  def steps_to_pct(self, pct=0.90):
    if not self.history: return None
    target = self.history[-1]['val_acc'] * pct
    for e in self.history:
      if e['val_acc'] >= target:
        return e['epoch']
    return None

  def summary(self):
    if not self.history: return
    f   = self.history[-1]
    ep1 = self.history[0]
    print(f"\n{'='*55}")
    print(f"  {self.name}")
    print(f"  CV @ epoch 1:     {ep1['cv']:.4f}")
    if len(self.history) >= 5:
      print(f"  CV @ epoch 5:  {self.history[4]['cv']:.4f}")
    print(f"  Final val acc:  {f['val_acc']*100:.2f}%")
    print(f"  Steps to 90%:   epoch {self.steps_to_pct()}")
    print(f"  Final util_H:   {f['util_entropy']:.4f}")
    if f['drift'] is not None:
      print(f"  Prototype drift:  {f['drift']:.4f}")
    print(f"  Total time:       {f['elapsed_min']:.1f} min")
    print(f"{'='*55}")

  def save(self, path):
    with open(path, 'w') as fh:
      json.dump({'name': self.name, 'history': self.history},
                                    fh, indent=2)
    print(f" Log → {path}")

  @classmethod
  def load(cls, path):
    with open(path) as fh:
        d = json.load(fh)
    obj = cls(d['name'])
    obj.history = d['history']
    return obj


# ────────────────────────────────────────────────
# SECTION 3 — Training and validation loops
# ────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, schedule, lr_sched, step,
                    scaler, device):
  model.train()
  total_loss = 0.0
  total_correct = total_n = 0
  for imgs, labels in loader:
    imgs, labels = imgs.to(device), labels.to(device)
    lam = schedule.get_lambda(step)
    with torch.amp.autocast(device_type='cuda'):
        logits, lb = model(imgs, lam=lam)
        # lb.mean() handles cases where lb might be a vector per batch
        loss = F.cross_entropy(logits, labels) + lb.mean()

    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    scaler.step(optimizer)
    scaler.update()

    lr_sched.step()

    total_loss += loss.item() * imgs.size(0)
    total_correct += (logits.argmax(1) == labels).sum().item()
    total_n += imgs.size(0)
    step += 1

  return {'loss': total_loss / total_n,
          'acc':  total_correct / total_n,
          'step': step}


@torch.no_grad()
def validate(model, loader, moe_block, schedule, step, device):
  model.eval()
  total_correct = total_n = 0
  all_metrics = []
  # ✓ Extract the raw module if wrapped in DataParallel
  raw_model = (model.module if isinstance(model, nn.DataParallel)
                                              else model)
  with torch.no_grad():
    for imgs, labels in loader:
      imgs, labels = imgs.to(device), labels.to(device)
      lam = schedule.get_lambda(step)
      logits, metrics = model(imgs, lam=lam)
      total_correct += (logits.argmax(1) == labels).sum().item()
      total_n += imgs.size(0)

      if metrics is not None and isinstance(metrics, dict):
        all_metrics.append(metrics)

  base = {'val_acc': total_correct / total_n}
  if all_metrics:
    base['cv'] = np.mean([m['cv'] for m in all_metrics])
    base['util_entropy'] = np.mean([m['util_entropy']
                                    for m in all_metrics])
    base['expert_counts']= np.sum( [m['expert_counts']
                                for m in all_metrics],axis=0).tolist()
  else:
    base['cv'] = 0.0; base['util_entropy'] = 1.0;
    base['expert_counts'] = []
  return base


# ───────────────────────────────────────────
# SECTION 4 — Tiny ImageNet data loaders
# ───────────────────────────────────────────

def get_tinyimagenet_loaders(data_dir, batch_size=256, num_workers=4):
  """
  Tiny ImageNet: 200 classes, 100K train images, 64×64 resolution.
  On Colab: mount Drive and point to your uploaded copy, or:
    !wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
    !unzip -q tiny-imagenet-200.zip

  IMPORTANT: val/ directory needs reorganizing before first use.
  Set reorganize=True on first run, False after.
  """
  import shutil
  mean = [0.485, 0.456, 0.406]
  std  = [0.229, 0.224, 0.225]

  train_tf = transforms.Compose([
      transforms.Resize(256),
      transforms.RandomCrop(224),
      transforms.RandomHorizontalFlip(),
      transforms.RandAugment(num_ops=2, magnitude=9),
      transforms.ToTensor(),
      transforms.Normalize(mean, std)
  ])
  val_tf = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean, std)
  ])

  train_dir = os.path.join(data_dir, 'train')
  val_dir = os.path.join(data_dir, 'val')
  val_org = os.path.join(data_dir, 'val_organized')

  # If dataset missing in scratch pad, download it
  if not os.path.exists(data_dir) or len(os.listdir(val_org)) < 200:
    print(f"  Dataset directory '{data_dir}' not found."
          f" Downloading via Standford CDN...")
    os.makedirs('/content', exist_ok=True)
    os.system('wget -q http://cs231n.stanford.edu/tiny-imagenet-200.zip -O /content/tiny-imagenet-200.zip')
    print("  Extracting archives into local runtime memory space...")
    os.system('unzip -q /content/tiny-imagenet-200.zip -d /content/')

  # reorganize val/ on first run
  if not os.path.exists(val_org):
    print("Reorganizing val/ into class subdirectories...")
    ann = os.path.join(val_dir, 'val_annotations.txt')
    with open(ann) as f:
      for line in f:
        parts = line.strip().split('\t')
        fname = parts[0]; cls = parts[1]
        src = os.path.join(val_dir, 'images', fname)
        dst = os.path.join(val_org, cls)
        os.makedirs(dst, exist_ok=True)
        shutil.copy(src, os.path.join(dst, fname))
    print(f"  Done → {val_org}")

  train_ds = datasets.ImageFolder(train_dir, train_tf)
  val_ds = datasets.ImageFolder(val_org,   val_tf)

  train_loader = DataLoader(train_ds, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers,
                            pin_memory=True, drop_last=True)
  val_loader = DataLoader(val_ds, batch_size=batch_size,
                          shuffle=False, num_workers=num_workers,
                          pin_memory=True)
  print(f" Tiny ImageNet: {len(train_ds):,} train | "
        f"{len(val_ds):,} val | {len(train_loader)} steps/epoch")
  return train_loader, val_loader


# ──────────────────────────────────
# SECTION 5 — Backbone loader
# ──────────────────────────────────

def load_deit(device='cpu'):
  """
  Load frozen DeiT-B/16 backbone (selected by H1).
  """
  print("Loading facebook/deit-base-patch16-224 ...")
  from transformers import ViTModel
  model = ViTModel.from_pretrained(
      "facebook/deit-base-patch16-224",
      output_attentions=True,
      attn_implementation="eager",
  )
  model.eval()
  for p in model.parameters():
      p.requires_grad = False
  return model.to(device)


def load_backbone_smoke(device='cpu'):
  """Random ViT-B/16 for smoke test (no download)."""
  config = ViTConfig(
      hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
      intermediate_size=3072, image_size=224, patch_size=16,
      num_channels=3, attn_implementation="eager"
  )
  model = ViTModel(config).eval()
  for p in model.parameters():
    p.requires_grad = False
  return model.to(device)


# ─────────────────────────────────────────
# SECTION 6 — Per-experiment runner
# ─────────────────────────────────────────

def run_experiment(exp_id, backbone, train_loader, val_loader,
                epochs=70, device='cpu', out_dir='.'):
  """
  Run one experiment (E1–E4) and return its ExperimentLogger.

  exp_id: 'E1', 'E2', 'E3', or 'E4'
  """
  print(f"\n{'─'*55}")
  print(f"  Running {exp_id}")
  print(f"{'─'*55}")

  total_steps = epochs * len(train_loader)
  num_classes = 200

  # ── Build model ──────
  if exp_id == 'E1':
    # Dense ViT-B: no MoE, standard FFN
    model      = DenseViT(backbone, num_classes).to(device)
    moe_block  = None
    schedule   = CosineAnnealSchedule(total_steps, frac=0.5)
    run_name   = 'E1_Dense_TinyImageNet'

  elif exp_id == 'E2':
    # V-MoE: learned linear router only
    moe_block  = StandardMoEBlock(E=4, k=2, lb=0.01).to(device)
    model      = WrappedViT(backbone, moe_block, num_classes).to(device)
    schedule   = CosineAnnealSchedule(total_steps, frac=0.5)
    run_name   = 'E2_VMoE_TinyImageNet_E4k2'

  elif exp_id == 'E3':
    # Expert Choice: inverted routing, no aux loss
    moe_block  = ExpertChoiceBlock(E=4, cf=2.0).to(device)
    model      = WrappedViT(backbone, moe_block, num_classes).to(device)
    schedule   = CosineAnnealSchedule(total_steps, frac=0.5)
    run_name   = 'E3_EC_TinyImageNet_E4'

  elif exp_id == 'E4':
    # Version B: W_r(z) + λ(t)·Pᵢ, T_anneal=50%
    moe_block  = AttnRouteMoEBlock(E=4, k=2, version='B',
                                    lb=0.01, gamma=0.001).to(device)
    model      = WrappedViT(backbone, moe_block, num_classes).to(device)
    schedule   = CosineAnnealSchedule(total_steps, frac=0.50)
    run_name   = 'E4_VersionB_TinyImageNet_E4k2_T50'

  elif exp_id == 'E4b':
    # Version B: W_r(z) + λ(t)·Pᵢ, T_anneal=25%
    moe_block  = AttnRouteMoEBlock(E=4, k=2, version='B',
                                    lb=0.01, gamma=0.001).to(device)
    model      = WrappedViT(backbone, moe_block, num_classes).to(device)
    schedule   = CosineAnnealSchedule(total_steps, frac=0.25)
    run_name   = 'E4b_VersionB_TinyImageNet_E4k2_T25'

  else:
    raise ValueError(f"Unknown exp_id: {exp_id}")

  # Wrap the entire model here so MoE layers run on both GPUs
  if device == 'cuda' and torch.cuda.device_count() > 1:
    print(f'Multi-GPU Active: Distributing across'
          f'{torch.cuda.device_count()} GPUs!')
    model = nn.DataParallel(model)

  # ── Optimizer ─────
  trainable = [p for p in model.parameters() if p.requires_grad]
  optimizer = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=0.05)
  lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=total_steps, eta_min=1e-6
  )
  logger = ExperimentLogger(run_name)
  step, best_acc = 0, 0.0
  ckpt_path = os.path.join(out_dir, f"{run_name}_best.pt")

  scaler = torch.amp.GradScaler('cuda')
  # ── Training loop ───────
  for epoch in range(1, epochs + 1):
    train_m = train_one_epoch(model, train_loader, optimizer,
                              schedule, lr_sched, step, scaler, device)
    step = train_m['step']

    val_m = validate(model, val_loader, moe_block, schedule,
                     step, device)
    lam = schedule.get_lambda(step)
    drift = (moe_block.prior.drift() if exp_id == 'E4'
                      and hasattr(moe_block, 'prior') else None)
    logger.log(epoch, train_m, val_m, lam, drift)

    # Save to disk periodically to avoid I/O bottlenecks:
    if epoch % 5 == 0 or epoch == epochs:
      logger.save(os.path.join(out_dir, f"{run_name}_log.json"))

    if val_m['val_acc'] > best_acc:
        best_acc = val_m['val_acc']
        torch.save({
            'epoch':   epoch,
            'val_acc': best_acc,
            'state':   model.state_dict(),
        }, ckpt_path)

    # keepalive every 10 epochs
    if epoch % 10 == 0:
        elapsed = logger.history[-1]['elapsed_min']
        print(f" [keepalive] epoch {epoch}/{epochs} | "
              f"best={best_acc*100:.1f}% | {elapsed:.0f} min elapsed")

  # ── Save and plot ──────
  drive_dir = '/content/drive/MyDrive/tinyimagenet_experiments'
  os.makedirs(drive_dir, exist_ok=True)

  final_log_path = os.path.join(drive_dir, f"{run_name}_log.json")
  logger.save(final_log_path)
  logger.summary()
  print(f"Saving individual performance graph to Google Drive...")
  _plot_single(logger, out_dir=drive_dir)

  return logger


# ──────────────────────────────────────────────────
# SECTION 7 — Per-experiment training curve plot
# ──────────────────────────────────────────────────

def _plot_single(logger, out_dir):
  """4-panel training curve for one experiment."""
  h    = logger.history
  eps  = [e['epoch']        for e in h]
  accs = [e['val_acc']*100  for e in h]
  cvs  = [e['cv']           for e in h]
  lams = [e['lambda']       for e in h]
  ues  = [e['util_entropy'] for e in h]

  fig, axes = plt.subplots(2, 2, figsize=(12, 7))
  fig.suptitle(logger.name, fontsize=11, fontweight='bold')

  axes[0,0].plot(eps, accs, 'b-', lw=2)
  axes[0,0].set_xlabel('Epoch'); axes[0,0].set_ylabel('Val acc (%)')
  axes[0,0].set_title('(a) Validation accuracy')
  axes[0,0].grid(alpha=0.3)

  axes[0,1].plot(eps, cvs, 'r-', lw=2)
  if cvs:
    ep1_idx = eps.index(1) if 1 in eps else 0
    axes[0,1].annotate(
        f"CV@ep1={cvs[ep1_idx]:.3f}", xy=(eps[ep1_idx], cvs[ep1_idx]),
        xytext=(10, 10),  # Offset by 10 pts R, 10 pts U
      textcoords="offset points", # Keep offst consist rgardlss axis scale
        fontsize=8, color='red')
  axes[0,1].set_xlabel('Epoch')
  axes[0,1].set_ylabel('CV (collapse rate)')
  axes[0,1].set_title('(b) Expert collapse rate  ← lower is better')
  axes[0,1].grid(alpha=0.3)

  axes[1,0].plot(eps, lams, 'g-', lw=2)
  axes[1,0].fill_between(eps, lams, alpha=0.15, color='green')
  axes[1,0].set_xlabel('Epoch'); axes[1,0].set_ylabel('λ(t)')
  axes[1,0].set_title('(c) Prior annealing schedule')
  axes[1,0].grid(alpha=0.3)

  axes[1,1].plot(eps, ues, 'm-', lw=2)
  axes[1,1].set_ylim(0, 1.05)
  axes[1,1].set_xlabel('Epoch')
  axes[1,1].set_ylabel('Util entropy (0–1)')
  axes[1,1].set_title('(d) Expert utilization  ← higher is better')
  axes[1,1].grid(alpha=0.3)

  plt.tight_layout()
  path = os.path.join(out_dir, f"{logger.name}_curves.png")
  plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
  print(f"  Curve → {path}")


# ─────────────────────────────────────────────────────────────────
# SECTION 8 — Figure 2: CV comparison across all four experiments
# ─────────────────────────────────────────────────────────────────

def make_figure2(logs, out_dir):
  """
  Figure 2 of the paper: CV curves for E1–E4 on the same axes.
  Also produces the companion accuracy overlay.

  What to look for:
    E4 (Version B) CV curve should be BELOW E2 (V-MoE) at epoch 1
    E3 (Expert Choice) CV should be near 0 throughout (by construction)
    E1 (Dense) has no routing — CV is 0 by definition

  If E4 CV@ep1 is NOT clearly below E2 CV@ep1: debug before E5–E12.
  """
  # Explicitly map short experiment IDs directly to their colors and paper names
  metadata = {
      'E1': {'color': '#888888', 'label': 'Dense ViT-B (no MoE)'},
      'E2': {'color': '#e41a1c', 'label': 'V-MoE'},
      'E3': {'color': '#377eb8', 'label': 'Expert Choice'},
      'E4': {'color': '#4daf4a', 'label': 'AttnRoute-MoE B'},
  }

  plt.rcParams.update({'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 11})
  fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

  lines_drawn = 0

  for exp_id, logger in logs.items():
    matched_key = None
    for k in metadata.keys():
        if k in exp_id:
            matched_key = k
            break

    if not matched_key:
        print(f"⚠️ Skipping unrecognized log key: {exp_id}")
        continue

    h    = logger.history
    eps  = [e['epoch']       for e in h]
    cvs  = [e['cv']          for e in h]
    accs = [e['val_acc']*100 for e in h]

    # Safely pull color and label configuration
    c     = metadata[matched_key]['color']
    label = metadata[matched_key]['label']

    axes[0].plot(eps, cvs,  color=c, lw=2.2, label=label)
    axes[1].plot(eps, accs, color=c, lw=2.2, label=label)
    lines_drawn += 1

  print(f"📊 Successfully plotted {lines_drawn} experimental runs onto axes.")

  axes[0].set_xlabel('Epoch')
  axes[0].set_ylabel('Expert collapse CV')
  axes[0].set_title('Expert Collapse Rate over training\n'
                  'Lower = less collapse, earlier = faster stabilization')
  if lines_drawn > 0: axes[0].legend(fontsize=9)
  axes[0].grid(True, linestyle='--', alpha=0.5)

  axes[1].set_xlabel('Epoch')
  axes[1].set_ylabel('Val accuracy (%)')
  axes[1].set_title('Validation accuracy\n'
                       'All methods should reach similar final accuracy')
  if lines_drawn > 0: axes[1].legend(fontsize=9)
  axes[1].grid(True, linestyle='--', alpha=0.5)

  plt.tight_layout()
  path = os.path.join(out_dir, 'week5_cv_comparison.png')
  plt.savefig(path, dpi=300, bbox_inches='tight')
  plt.close()
  print(f"\n  Figure 2 → {path}")
  return path


# ───────────────────────────────────────────
# SECTION 9 — Table 2 Tiny ImageNet builder
# ───────────────────────────────────────────

def make_table2_tinyimagenet(logs, out_dir):
  """
  Print and save the TinyImageNet portion of Table 2.
  Paste the LaTeX output directly into tables/table2_filled.tex.

  Expected values for a well-functioning run:
    Dense: CV=0, acc ~75-80%
    V-MoE: CV@ep1 ~0.15-0.35, final acc ~75-80%
    EC:    CV@ep1 ~0.00, final acc ~75-80%
    VB:    CV@ep1 LOWER than V-MoE, steps-90% FEWER than V-MoE
  """
  rows = []
  method_order = ['E1', 'E2', 'E3', 'E4']
  method_names = {
    'E1': r'Dense ViT-B',
    'E2': r'V-MoE~\citep{riquelme2021vmoe}',
    'E3': r'Expert Choice~\citep{zhou2022expertchoice}',
    'E4': r'AttnRoute-MoE B \textbf{(ours)}',
  }

  for exp_id in method_order:
    if exp_id not in logs:
        continue
    logger = logs[exp_id]
    h      = logger.history
    if not h: continue

    f    = h[-1]
    ep1  = h[0]
    ep5  = h[4]['cv'] if len(h) > 4 else None
    s90  = logger.steps_to_pct(0.90)

    rows.append({
        'method':    method_names[exp_id],
        'acc':       f['val_acc'] * 100,
        'cv_ep1':    ep1['cv'],
        'cv_ep5':    ep5,
        'cv_fin':    f['cv'],
        'util_ep1':  ep1['util_entropy'],
        'steps_90':  s90,
    })

  # ── plain text summary ──
  print(f"\n{'═'*80}")
  print("TABLE 2 — Tiny ImageNet results")
  print(f"{'Method':<35} {'Acc':>5} {'CV@1':>7} {'CV@5':>7} "
          f"{'CVfin':>6} {'H@1':>6} {'Steps90':>8}")
  print('─'*80)
  for r in rows:
    ep5_s = f"{r['cv_ep5']:.4f}" if r['cv_ep5'] is not None else "  —   "
    print(f"{r['method'].replace(chr(92), '').replace('{', '').replace('}', ''):<35} "
          f"{r['acc']:>5.1f} {r['cv_ep1']:>7.4f} {ep5_s:>7} "
          f"{r['cv_fin']:>6.4f} {r['util_ep1']:>6.4f} "
          f"{str(r['steps_90']):>8}")
    print('═'*80)

  # ── OBSERVE block ────
  if len(rows) >= 4:
    vb  = next((r for r in rows if 'ours' in r['method']), None)
    vm  = next((r for r in rows if 'V-MoE' in r['method']), None)
    ec  = next((r for r in rows if 'Expert' in r['method']), None)
    if vb and vm:
      ratio = (vm['cv_ep1'] - vb['cv_ep1']) / (vm['cv_ep1'] + 1e-8) * 100
      print(f"\n OBSERVE:")
      print(f" CV@ep1 reduction vs V-MoE: {ratio:+.1f}%")
      if ratio < 5:
        print(f" ✗ LESS THAN 5% — debug before running Phrase 2 sweeps.")
        print(f"1. Print Pᵢ.std() in first batch — want > 0.1")
        print(f"2. Confirm λ(step=0) = 1.0 (not 0.0)")
        print(f"3. Confirm backbone is frozen (no backbone grads)")
        print(f"4. Try T_anneal_frac=0.75 instead of 0.50")
        print(f"5. Try lb_coeff=0.0 for E2 — higher lb may mask collapse")
      elif ratio < 10:
        print(f"⚠ 5–10% — borderline. Proceed but note in paper.")
      else:
        print(f" ✓ {ratio:.1f}% improvement. Proceed safely to Phase 2 sweeps.")
      if vb['steps_90'] and vm['steps_90']:
        step_diff = vm['steps_90'] - vb['steps_90']
        print(f"Steps-to-90% advantage: {step_diff:+d} epochs "
              f"({'faster' if step_diff > 0 else 'slower'})")
      if ec:
        if ec['cv_ep1'] > 0.05:
          print(f" ✗ Expert Choice CV@ep1={ec['cv_ep1']:.4f} > 0.05 "
                f"— check capacity_factor implementation")
        else:
          print(f" ✓ Expert Choice CV@ep1≈{ec['cv_ep1']:.4f} "
                f"(balanced by construction)")
      acc_diff = abs(vb['acc'] - vm['acc'])
      if acc_diff > 2.0:
        print(f" ⚠ Acc gap={acc_diff:.1f}% (>2%) — "
              f"MoE should match Dense within ~1%. "
              f"Check lb_coeff or reduce E.")

  # ── LaTeX output ──────
  latex_lines = [r"\begin{table}[h]", r"\centering",
    r"\caption{TinyImageNet Phase 1 MoE Performance profiles}",
    r"\begin{tabular}{lccccc}", r"\toprule",
    r"Method & Top-1 (\%) & CV@ep1 $\downarrow$ & CV@ep5 & Steps-90\% & Util-H@ep1 \\",
    r"\midrule"]
  for r in rows:
    ep5_s = f"{r['cv_ep5']:.4f}" if r['cv_ep5'] is not None else "---"
    latex_lines.append(
        f"{r['method']} & {r['acc']:.1f} & {r['cv_ep1']:.4f} & "
        f"{ep5_s} & {r['steps_90']} & {r['util_ep1']:.4f} \\\\"
    )
  latex_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
  latex_str = '\n'.join(latex_lines)

  tbl_path = os.path.join(out_dir, 'week5_table2_tinyimagenet.txt')
  with open(tbl_path, 'w') as fh:
      fh.write(latex_str)
  print(f"\n  Table 2 LaTeX → {tbl_path}")
  print("Copy this into tables/table2_filled.tex and \\input it in main.tex\n")
  return latex_str


# ──────────────────────────────────────────────────────
# SECTION 10 — Smoke test 
# ─────────────────────────────────────────────────────

def smoke_test(out_dir='.'):
  print('='*60)
  print('WEEK 5 SMOKE TEST')
  print('='*60)

  device = 'cpu'
  os.makedirs(out_dir, exist_ok=True)

  backbone = load_backbone_smoke(device)

  # tiny fake CIFAR-like dataset: 32 images
  fake_imgs   = torch.randn(32, 3, 224, 224)
  fake_labels = torch.randint(0, 100, (32,))
  loader = DataLoader(TensorDataset(fake_imgs, fake_labels),
                      batch_size=8, shuffle=True)

  logs = {}
  for exp_id in ['E1', 'E2', 'E3', 'E4']:
      print(f"\n── {exp_id} smoke ───────")
      logger = run_experiment(exp_id, backbone, loader, loader,
                            epochs=3, device=device, out_dir=out_dir)
      logs[exp_id] = logger

  print('\n── Figure 2 ───────')
  make_figure2(logs, out_dir)

  print('\n── Table 2 TinyImage Net ────────')
  make_table2_tinyimagenet(logs, out_dir)

  stop_or_proceed(logs)

  print(f'\n{"="*60}')
  print('SMOKE TEST COMPLETE')
  expected = ['week5_cv_comparison.png',
              'week5_table2_tinyimagenet.txt']
  for f in expected:
    path = os.path.join(out_dir, f)
    status = '✓' if os.path.exists(path) else '✗'
    print(f'  {status} {f}')
  print('='*60)


# ──────────────────────────────────
# SECTION 12 — Full run (Colab)
# ──────────────────────────────────

def full_run(exp_id, out_dir='',
        data_dir='/content/tiny-imagenet-200', epochs=70, device=None):
  """
  Run one experiment on Kaggle. Call once per experiment ID.
  Saves log JSON and training curve PNG to out_dir.
  """
  if device is None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
  os.makedirs(out_dir, exist_ok=True)
  print(f'Device: {device}')
  if device == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')

  backbone     = load_deit(device)
  train_loader, val_loader = get_tinyimagenet_loaders(data_dir,
                                                batch_size=256)
  logger = run_experiment(exp_id, backbone, train_loader, val_loader,
                      epochs=epochs, device=device, out_dir=out_dir)
  return logger


def compare_mode(log_dir='', out_dir=''):
  """
  Load all saved JSON logs and produce Figure 2 + Table 2.
  Run after all four experiments are complete.
  """
  id_prefixes = {
      'E1': 'E1_Dense',
      'E2': 'E2_VMoE',
      'E3': 'E3_EC',
      'E4': 'E4_VersionB',
  }
  logs = {}
  for exp_id, prefix in id_prefixes.items():
    # find the log file
    for fname in os.listdir(log_dir):
      if fname.startswith(prefix) and fname.endswith('_log.json'):
        path = os.path.join(log_dir, fname)
        logs[exp_id] = ExperimentLogger.load(path)
        print(f"  Loaded {exp_id}: {fname}")
        break
    if exp_id not in logs:
      print(f"  ✗ {exp_id} log not found in {log_dir}")

  if logs:
    print("\nCompiling comparative results directly to Google Drive...")
    drive_dir = '/content/drive/MyDrive/tinyimagenet_experiments'

    make_figure2(logs, out_dir=drive_dir)
    make_table2_tinyimagenet(logs, out_dir=drive_dir)
    print(f"🏁 All multi-experiment charts and matrices successfully saved to Google Drive!")


# ────────────────────────────────
# ENTRY POINT
# ────────────────────────────────
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--mode', choices=['smoke', 'full', 'compare'],
                      default='full')
  parser.add_argument('--exp', default='E4',
                  help='E1/E2/E3/E4/E4b')
  parser.add_argument('--out_dir', default='/content/outputs')
  parser.add_argument('--data_dir', default='/content/tiny-imagenet-200')
  parser.add_argument('--epochs', type=int, default=45)
  args, _ = parser.parse_known_args()

  os.makedirs(args.out_dir, exist_ok=True)

  if args.mode == 'smoke':
    smoke_test(out_dir=args.out_dir)
  elif args.mode == 'full':
    full_run(args.exp, out_dir=args.out_dir,
            data_dir=args.data_dir, epochs=args.epochs)
  else:
    drive_dir = '/content/drive/MyDrive/tinyimagenet_experiments'
    # Check if the folder is currently visible to your environment
    if os.path.exists(drive_dir):
      print(f"📂 Drive folder found. Reading experimental results directly from: {drive_dir}")
      compare_mode(log_dir=drive_dir, out_dir=drive_dir)
    else:
      print(f"⚠️ Notice: Drive path '{drive_dir}' not detected via virtual filesystem.")
      print("Attempting a safe mount to fetch logs...")
      try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=True)
        compare_mode(log_dir=drive_dir, out_dir=drive_dir)
      except Exception as e:
        print(f"❌ Could not access Drive directory."
              f"Falling back to local directory ({args.out_dir}). Error: {e}")
        compare_mode(log_dir=args.out_dir, out_dir=args.out_dir)
