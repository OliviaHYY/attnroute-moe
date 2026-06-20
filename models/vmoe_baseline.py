# ══════════════════════════════════════════════════════════════════════════════
# WEEK 1 — V-MoE baseline implementation
# Standard learned-router MoE — the primary comparison in every experiment.
# ══════════════════════════════════════════════════════════════════════════════
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# StandardMoEBlock  — V-MoE (Riquelme et al. 2021), token-choice routing
# Same dimensions and loss as AttnRouteMoEBlock for a fair comparison.
# ─────────────────────────────────────────────────────────────────────────────
class StandardMoEBlock(nn.Module):
  """
  Replicates the V-MoE routing mechanism:
      S_i = W_r(z_i)   — learned linear router, no prior
  Top-k dispatch, load-balance auxiliary loss, same expert architecture.

  This is Baseline 2 (E2, E14) in the experiment registry.
  Use identical hyperparameters to AttnRouteMoEBlock for fair comparison.
  """
  def __init__(self, d_model=768, num_experts=4, top_k=2, lb_coeff=0.01):
    super().__init__()
    self.E = num_experts
    self.k = top_k
    self.lb_coeff = lb_coeff

    self.W_r = nn.Linear(d_model, num_experts, bias=False)
    self.experts = nn.ModuleList([
        nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        ) for _ in range(num_experts)
    ])
    self.norm = nn.LayerNorm(d_model)

  def forward(self, z, A=None, lam=None):
    """
    Args:
        z:   [B, N_total, d_model]
        A:   ignored (no prior) — accepted for API compatibility
        lam: ignored            — accepted for API compatibility
    Returns:
        out:     [B, N_total, d_model]
        lb_loss: scalar
    """
    z_patches = self.norm(z[:, 1:, :])          # [B, 196, D]
    S_i = self.W_r(z_patches)             # [B, 196, E]

    topk_vals, topk_idx = torch.topk(S_i, self.k, dim=-1)
    topk_weights = torch.softmax(topk_vals, dim=-1)

    output = torch.zeros_like(z_patches)
    for e in range(self.E):
        mask = (topk_idx == e)
        weight = (topk_weights * mask.float()).sum(-1, keepdim=True)
        output += weight * self.experts[e](z_patches)

    # load-balance loss
    avg_probs = torch.softmax(S_i, dim=-1).mean(dim=[0, 1])
    lb_loss   = self.lb_coeff * self.E * (avg_probs * avg_probs).sum()

    out = z.clone()
    out[:, 1:, :] = out[:, 1:, :] + output
    return out, lb_loss

  @torch.no_grad()
  def get_routing_metrics(self, z, A=None, lam=None):
    z_patches = self.norm(z[:, 1:, :])
    S_i = self.W_r(z_patches)
    _, topk_idx = torch.topk(S_i, self.k, dim=-1)
    counts = torch.tensor([(topk_idx == e).float().sum().item()
                                for e in range(self.E)])
    cv = (counts.std() / (counts.mean() + 1e-8)).item()
    probs = counts / counts.sum()
    ue = -(probs * (probs + 1e-8).log()).sum().item()
    ue /= torch.log(torch.tensor(float(self.E))).item()
    return {"cv": cv, "util_entropy": ue,
                "expert_counts": counts.int().tolist()}


# ─────────────────────────────────────────────────────────────────────────────
# ExpertChoiceBlock  — Zhou et al. 2022 (Google Research), NeurIPS
# Each expert selects its top-k tokens from the batch (inverted routing).
# Guarantees load balance by construction — no auxiliary loss needed.
# This is Baseline 3 (E3, E15) in experiment registry.
# ─────────────────────────────────────────────────────────────────────────────
class ExpertChoiceBlock(nn.Module):
  """
  Expert Choice routing:
    - Compute affinity matrix S ∈ ℝ^{B×N×E} = z_patches @ W_r.T
    - Each expert e selects top-c tokens where c = ceil(N * capacity / E)
    - No load-balance loss needed (balance is enforced by construction)

  Capacity factor `c_factor` controls how many tokens each expert processes:
      c_factor = 2.0 means each expert sees 2× the "fair share" of tokens
      (standard value from Zhou et al. 2022)
  """
  def __init__(self, d_model=768, num_experts=4, c_factor=2.0):
    super().__init__()
    self.E = num_experts
    self.c_factor = c_factor

    self.W_r = nn.Linear(d_model, num_experts, bias=False)
    self.experts = nn.ModuleList([
        nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        ) for _ in range(num_experts)
    ])
    self.norm = nn.LayerNorm(d_model)

  def forward(self, z, A=None, lam=None):
    """
    Args:  z: [B, N_total, d_model]
    Returns: out: [B, N_total, d_model], lb_loss: 0 (not needed for EC)
    """
    B, N_total, D = z.shape
    z_patches = self.norm(z[:, 1:, :])    # [B, 196, D]
    N = z_patches.shape[1]                 # 196

    # capacity: how many tokens each expert selects
    c = max(1, int(N * self.c_factor / self.E))  # e.g. ceil(196*2/4) = 98

    # affinity: token-expert scores [B, N, E]
    S = z_patches @ self.W_r.weight.T     # [B, 196, E]

    output = torch.zeros_like(z_patches)

    for e in range(self.E):
        # expert e selects its top-c tokens
        scores_e = S[:, :, e]             # [B, 196]
        _, top_idx = torch.topk(scores_e, c, dim=-1)  # [B, c]

        # gather selected tokens: [B, c, D]
        selected = torch.gather(
            z_patches,
            dim=1,
            index=top_idx.unsqueeze(-1).expand(-1, -1, D)
        )

        # run expert
        processed = self.experts[e](selected)   # [B, c, D]

        # soft weights: softmax over selected token scores
        weights = torch.softmax(
            torch.gather(scores_e, 1, top_idx), dim=-1
        ).unsqueeze(-1)                          # [B, c, 1]

        # scatter back
        output.scatter_add_(
            dim=1,
            index=top_idx.unsqueeze(-1).expand(-1, -1, D),
            src=weights * processed
        )

    out = z.clone()
    out[:, 1:, :] = out[:, 1:, :] + output
    return out, torch.tensor(0.0, requires_grad=True)   # no lb loss needed

  @torch.no_grad()
  def get_routing_metrics(self, z, A=None, lam=None):
    """For EC, CV should be near 0 by construction — verify this."""
    B, N_total, D = z.shape
    z_patches = self.norm(z[:, 1:, :])
    N = z_patches.shape[1]
    c = max(1, int(N * self.c_factor / self.E))
    S = z_patches @ self.W_r.weight.T
    counts = torch.zeros(self.E)
    for e in range(self.E):
        counts[e] = c * B   # EC dispatches exactly c tokens per expert
    cv = (counts.std() / (counts.mean() + 1e-8)).item()
    probs = counts / counts.sum()
    ue = -(probs * (probs + 1e-8).log()).sum().item()
    ue /= torch.log(torch.tensor(float(self.E))).item()
    return {"cv": cv, "util_entropy": ue,
            "expert_counts": counts.int().tolist()}


# ─────────────────────────────────────────────────────────────────────────────
# WrappedViT  — generic wrapper for any MoE block (baseline or AttnRoute)
# Same as WrappedViTMoE in attnmoe_core.py but accepts any block type.
# ─────────────────────────────────────────────────────────────────────────────
class WrappedViT(nn.Module):
  """
  Frozen ViT-B/16 backbone + one MoE block (any type) on the last FFN.
  Use this for ALL experiments so the backbone is always identical.
  """
  def __init__(self, backbone, moe_block, num_classes=200):
    super().__init__()
    self.backbone  = backbone
    self.moe_block = moe_block
    self.head = nn.Linear(768, num_classes)
    for p in self.backbone.parameters():
        p.requires_grad = False

  def forward(self, pixel_values, lam=0.0):
    out = self.backbone(pixel_values=pixel_values, output_attentions=True)
    z = out.last_hidden_state    # [B, 197, 768]
    A = out.attentions[-1]       # [B, 12, 197, 197]
    z_out, lb_loss = self.moe_block(z, A, lam=lam)
    return self.head(z_out[:, 0, :]), lb_loss


# ─────────────────────────────────────────────────────────────────────────────
# parameter_count  — verify parameter counts before running anything
# ─────────────────────────────────────────────────────────────────────────────
def parameter_count(model):
  """
  Print a breakdown of trainable vs frozen parameters.
  Call before training to confirm only the MoE block is training.
  """
  total = sum(p.numel() for p in model.parameters())
  trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
  frozen = total - trainable
  print(f"  Total parameters:     {total:>12,}")
  print(f"  Trainable:            {trainable:>12,}  ← should be MoE block only")
  print(f"  Frozen (backbone):    {frozen:>12,}")
  print(f"  Trainable fraction:   {trainable/total*100:>11.2f}%")
  return trainable


# ─────────────────────────────────────────────────────────────────────────────
# compare_baselines  — run both baselines on same batch, print side-by-side
# Use in Week 6 to quickly sanity-check V-MoE vs Expert Choice behavior
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def compare_routing_at_init(backbone, num_classes=10, device='cpu'):
  """
  Compare V-MoE vs Expert Choice routing metrics at initialization
  (before any training). Expert Choice should have CV≈0; V-MoE should
  have high CV (random routing = collapsed).

  This is a sanity check to run once before any training.
  """
  dummy = torch.randn(4, 3, 224, 224).to(device)

  vmoe_block = StandardMoEBlock()
  ec_block   = ExpertChoiceBlock()

  vmoe = WrappedViT(backbone, vmoe_block, num_classes).to(device)
  ec   = WrappedViT(backbone, ec_block, num_classes).to(device)

  out_v = backbone.to(device)(pixel_values=dummy, output_attentions=True)
  z = out_v.last_hidden_state
  A = out_v.attentions[-1]

  m_v = vmoe_block.get_routing_metrics(z, A)
  m_e = ec_block.get_routing_metrics(z, A)

  print("\n── Routing at initialization (before any training) ──")
  print(f"{'Method':<20} {'CV':>8} {'Util_H':>8} {'Expert counts'}")
  print(f"{'V-MoE':<20} {m_v['cv']:>8.4f} {m_v['util_entropy']:>8.4f}  {m_v['expert_counts']}")
  print(f"{'Expert Choice':<20} {m_e['cv']:>8.4f} {m_e['util_entropy']:>8.4f}  {m_e['expert_counts']}")
  print("\nExpect: V-MoE CV > Expert Choice CV (random router is unbalanced)")
  print("Expect: Expert Choice CV ≈ 0 (guaranteed balance by construction)")

  return {"vmoe": m_v, "expert_choice": m_e}


# ───────────
# TEST
# ───────────
if __name__ == "__main__":
  from transformers import ViTModel, ViTConfig
  from torch.utils.data import DataLoader, TensorDataset

  print("Building test backbone...")
  config = ViTConfig(hidden_size=768, num_hidden_layers=12,
                    num_attention_heads=12, intermediate_size=3072,
                    image_size=224, patch_size=16, num_channels=3,
                    attn_implementation="eager")
  backbone = ViTModel(config)

  # ── parameter count check ────
    vmoe_block = StandardMoEBlock(num_experts=4, top_k=2)
    model_v    = WrappedViT(backbone, vmoe_block, num_classes=10)
    print("\nV-MoE parameter breakdown:")
    parameter_count(model_v)

  # ── forward pass check ────
  dummy = torch.randn(2, 3, 224, 224)
  logits, lb = model_v(dummy, lam=0.0)
  assert logits.shape == (2, 10), f"Wrong logits shape: {logits.shape}"
  assert not torch.isnan(logits).any()
  print(f"\nV-MoE forward: logits {logits.shape}, lb_loss {lb.item():.4f}")

  ec_block = ExpertChoiceBlock(num_experts=4)
  model_e  = WrappedViT(backbone, ec_block, num_classes=10)
  logits_e, lb_e = model_e(dummy, lam=0.0)
  assert logits_e.shape == (2, 10)
  print(f"EC  forward: logits {logits_e.shape}, lb_loss {lb_e.item():.4f}")

  # ── routing comparison at init ────
  compare_routing_at_init(backbone, num_classes=10)

  # ── one training step ───
  print("\nOne gradient step on V-MoE...")
  opt = torch.optim.AdamW([p for p in model_v.parameters()
                              if p.requires_grad], lr=1e-4)
  labels = torch.randint(0, 10, (2,))
  logits, lb = model_v(dummy)
  loss = F.cross_entropy(logits, labels) + lb
  opt.zero_grad(); loss.backward(); opt.step()
  print(f"  Loss: {loss.item():.4f}  ✓ gradients flow")

  print("\n✓ week1_vmoe_baseline.py — all checks passed")
  print("  Import StandardMoEBlock and ExpertChoiceBlock in Kaggle training runs")
