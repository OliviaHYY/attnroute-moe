import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import ViTModel, ViTConfig
import numpy as np

class AttentionPrior(nn.Module):
  def __init__(self, num_experts=4, eps=1e-8):
    super().__init__()
    self.E = num_experts
    self.eps = eps
    init = torch.zeros(num_experts, 3)
    for j in range(min(num_experts, 3)):
        init[j, j] = 1.0
    if num_experts > 3:
        init[3] = torch.tensor([1/3, 1/3, 1/3])
    self.prototypes = nn.Parameter(init)

  def forward(self, A):
    B, H, N, _ = A.shape
    A_patches = A[:, :, 1:, :]
    H_i = -(A_patches * (A_patches + self.eps).log()).sum(-1).mean(1)
    A_mean = A_patches.mean(1, keepdim=True)
    Var_i = ((A_patches - A_mean)**2).mean(-1).mean(1)
    A_cls_avg = A[:, :, 0, :].mean(1)
    A_patch_avg = A_patches.mean(1)
    cls_norm = F.normalize(A_cls_avg, dim=-1)
    patch_norm = F.normalize(A_patch_avg, dim=-1)
    C_i = (patch_norm * cls_norm.unsqueeze(1)).sum(-1)
    f_i = torch.stack([-H_i, Var_i, C_i], dim=-1)
    mean = f_i.mean(dim=[0,1], keepdim=True)
    std  = f_i.std(dim=[0,1], keepdim=True) + self.eps
    f_i  = (f_i - mean) / std
    return f_i @ self.prototypes.T


class ExpertFFN(nn.Module):
  def __init__(self, d_model=768):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(d_model, d_model * 4), nn.GELU(),
        nn.Linear(d_model * 4, d_model),
    )
  def forward(self, x):
    return self.net(x)


class AttnRouteMoEBlock(nn.Module):
  def __init__(self, d_model=768, num_experts=4, top_k=2,
                 version='B', lb_coeff=0.01):
    super().__init__()
    self.E = num_experts; self.k = top_k
    self.version = version; self.lb_coeff = lb_coeff
    self.prior   = AttentionPrior(num_experts)
    self.experts = nn.ModuleList([ExpertFFN(d_model) for _ in range(num_experts)])
    if version == 'B':
        self.W_r = nn.Linear(d_model, num_experts, bias=False)
    self.norm = nn.LayerNorm(d_model)

  def forward(self, z, A, lam=0.0):
    z_patches = self.norm(z[:, 1:, :])
    P_i = self.prior(A)
    if self.version == 'B':
        S_i = self.W_r(z_patches) + lam * P_i
    else:   # Version A 
        S_i = P_i
    topk_vals, topk_idx = torch.topk(S_i, self.k, dim=-1)
    topk_weights = torch.softmax(topk_vals, dim=-1)
    output = torch.zeros_like(z_patches)
    for e in range(self.E):
        mask = (topk_idx == e)
        weight = (topk_weights * mask.float()).sum(-1, keepdim=True)
        output += weight * self.experts[e](z_patches)
    router_probs = torch.softmax(S_i, dim=-1)
    avg_probs = router_probs.mean(dim=[0,1])
    lb_loss = self.lb_coeff * self.E * (avg_probs * avg_probs).sum()
    out = z.clone()
    out[:, 1:, :] = out[:, 1:, :] + output
    return out, lb_loss

  @torch.no_grad()
  def get_routing_metrics(self, z, A, lam=0.0):
    z_patches = self.norm(z[:, 1:, :])
    P_i = self.prior(A)
    S_i = (self.W_r(z_patches) + lam * P_i) if self.version == 'B' else P_i
    _, topk_idx = torch.topk(S_i, self.k, dim=-1)
    counts = torch.tensor([(topk_idx == e).float().sum().item() for e in range(self.E)])
    cv = (counts.std() / (counts.mean() + 1e-8)).item()
    probs = counts / counts.sum()
    ue = -(probs * (probs + 1e-8).log()).sum().item()
    ue /= torch.log(torch.tensor(float(self.E))).item()
    return {"cv": cv, "util_entropy": ue, "expert_counts": counts.int().tolist()}


class CosineAnnealSchedule:
  def __init__(self, total_steps, T_anneal_frac=0.5):
      self.T_anneal = int(total_steps * T_anneal_frac)
  def get_lambda(self, step):
      if step >= self.T_anneal: return 0.0
      import math
      return math.cos(math.pi * step / (2 * self.T_anneal)) ** 2


class WrappedViTMoE(nn.Module):
  def __init__(self, backbone, moe_block, num_classes=200):
    super().__init__()
    self.backbone = backbone
    self.moe_block = moe_block
    self.classifier = nn.Linear(768, num_classes)
    for p in self.backbone.parameters():
        p.requires_grad = False

  def forward(self, pixel_values, lam=0.0):
    outputs = self.backbone(pixel_values=pixel_values, output_attentions=True)
    z = outputs.last_hidden_state
    A = outputs.attentions[-1]
    z_out, lb_loss = self.moe_block(z, A, lam=lam)
    logits = self.classifier(z_out[:, 0, :])
    return logits, lb_loss


class ExperimentLogger:
  def __init__(self, name):
    self.name = name; self.history = []

  def log(self, epoch, train_m, val_m, lam):
    self.history.append({
        "epoch": epoch, "lambda": lam,
        "train_loss": train_m["avg_loss"], "train_acc": train_m["avg_acc"],
        "val_acc": val_m["val_acc"], "cv": val_m["cv"],
        "util_entropy": val_m["util_entropy"],
        "expert_counts": val_m["expert_counts"],
    })
    print(f" Ep {epoch:3d} | λ={lam:.3f} | loss={train_m['avg_loss']:.4f} | "
          f"acc={val_m['val_acc']*100:.1f}% | CV={val_m['cv']:.4f} | "
          f"H={val_m['util_entropy']:.3f} | experts={val_m['expert_counts']}")

  def steps_to_pct_accuracy(self, pct=0.90):
    final = self.history[-1]["val_acc"]
    for e in self.history:
        if e["val_acc"] >= final * pct: return e["epoch"]

  def summary(self):
    f = self.history[-1]; s90 = self.steps_to_pct_accuracy()
    print(f"\n{'='*50}")
    print(f" {self.name} | CV@ep1={self.history[0]['cv']:.4f} | "
          f"finalAcc={f['val_acc']*100:.2f}% | steps90={s90}")
    print(f"{'='*50}")

  def save(self, path):
    import json
    with open(path,'w') as fh: json.dump({"name":self.name,"history":self.history},fh,indent=2)
    print(f"Saved → {path}")


# ── training and validation functions ────
def train_one_epoch(model, loader, optimizer, schedule, current_step, device):
  model.train()
  total_loss = total_correct = total_samples = 0
  for imgs, labels in loader:
    imgs, labels = imgs.to(device), labels.to(device)
    lam = schedule.get_lambda(current_step)
    logits, lb_loss = model(imgs, lam=lam)
    loss = F.cross_entropy(logits, labels) + lb_loss
    optimizer.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    total_loss += loss.item() * imgs.size(0)
    total_correct += (logits.argmax(1) == labels).sum().item()
    total_samples += imgs.size(0)
    current_step  += 1
  return {"avg_loss": total_loss/total_samples, "avg_acc": total_correct/total_samples, "step": current_step}


@torch.no_grad()
def validate(model, loader, moe_block, schedule, step, device):
  model.eval()
  total_correct = total_samples = 0; all_metrics = []
  for imgs, labels in loader:
    imgs, labels = imgs.to(device), labels.to(device)
    lam = schedule.get_lambda(step)
    logits, _ = model(imgs, lam=lam)
    total_correct += (logits.argmax(1) == labels).sum().item()
    total_samples += imgs.size(0)
    outputs = model.backbone(pixel_values=imgs, output_attentions=True)
    z = outputs.last_hidden_state; A = outputs.attentions[-1]
    all_metrics.append(moe_block.get_routing_metrics(z, A, lam))
  return {
      "val_acc":  total_correct/total_samples,
      "cv":  np.mean([m["cv"] for m in all_metrics]),
      "util_entropy":  np.mean([m["util_entropy"] for m in all_metrics]),
      "expert_counts":  np.sum([m["expert_counts"] for m in all_metrics], axis=0).tolist(),
  }


# ── smoke test ──
print("Smoke test — 3 epochs on synthetic data...")
config = ViTConfig(hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
                   intermediate_size=3072, image_size=224, patch_size=16,
                   num_channels=3, attn_implementation="eager")
backbone = ViTModel(config)
moe = AttnRouteMoEBlock(version='B')
model = WrappedViTMoE(backbone, moe, num_classes=10)

fake_imgs = torch.randn(8, 3, 224, 224)
fake_labels = torch.randint(0, 10, (8,))
loader = DataLoader(TensorDataset(fake_imgs, fake_labels), batch_size=4)

optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
schedule = CosineAnnealSchedule(total_steps=100, T_anneal_frac=0.5)
logger = ExperimentLogger("smoke_test")

step = 0
for epoch in range(1, 4):
  tm = train_one_epoch(model, loader, optimizer, schedule, step, 'cpu')
  vm = validate(model, loader, moe, schedule, tm["step"], 'cpu')
  step = tm["step"]
  logger.log(epoch, tm, vm, schedule.get_lambda(step))

logger.summary()
# Personal_Save (change to your own route)
# logger.save("/home/claude/month1/smoke_test_log.json")
print("\n✓ Full training loop end-to-end")
