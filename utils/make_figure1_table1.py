# ══════════════════════════════════════════════════════════════════════════════
# H1 signal computation and backbone comparison
# Computes entropy, head variance, CLS-similarity for any ViT backbone.
# ══════════════════════════════════════════════════════════════════════════════
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image

# ── Image loading ────
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

image_transform = T.Compose([
  T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD),
])

def load_image(path):
  """Returns [1, 3, 224, 224] normalized tensor."""
  return image_transform(Image.open(path).convert("RGB")).unsqueeze(0)

def image_for_display(path):
  """Returns [224, 224, 3] float array in [0,1] for imshow."""
  img = Image.open(path).convert("RGB").resize((224, 224))
  return np.array(img).astype(np.float32) / 255.0

# ── Core: extract attention from a specific layer ───
def extract_attention(model, pixel_values, layer=11):
  """Returns A: [B, H, N, N] from encoder layer `layer` (0-indexed)."""
  with torch.no_grad():
      out = model(pixel_values=pixel_values, output_attentions=True)
  return out.attentions[layer]

# ── Core: three routing signals from A ───
def compute_routing_signals(A, eps=1e-8):
  """
  Args:  A: [B, H, 197, 197]
  Returns: H_i [B,196], Var_i [B,196], C_i [B,196]
  """
  A_patches = A[:, :, 1:, :]                               # [B,H,196,197]

  # 1. Entropy (avg across heads)
  H_i = -(A_patches * (A_patches + eps).log()).sum(-1).mean(1)  # [B,196]

  # 2. Cross-head variance
  Var_i = ((A_patches - A_patches.mean(1, keepdim=True))**2).mean(-1).mean(1) # [B,196]

  # 3. CLS cosine similarity
  A_cls   = F.normalize(A[:, :, 0, :].mean(1),  dim=-1)   # [B,197]
  A_patch = F.normalize(A_patches.mean(1), dim=-1)   # [B,196,197]
  C_i = (A_patch * A_cls.unsqueeze(1)).sum(-1)         # [B,196]

  return H_i, Var_i, C_i

# ── Core: measure foreground vs. background entropy gap ───
def measure_h1(H_i_single, mask_14x14):
  """
  Args:
      H_i_single:  [196] tensor — entropy for one image
      mask_14x14:  [14,14] BoolTensor — True = foreground
  Returns:
      dict(mean_fg, mean_bg, gap) or None if mask is degenerate
  """
  flat = mask_14x14.flatten()
  if flat.sum() == 0 or (~flat).sum() == 0:
      return None
  return {
      "mean_fg": H_i_single[flat].mean().item(),
      "mean_bg": H_i_single[~flat].mean().item(),
      "gap":     H_i_single[~flat].mean().item() - H_i_single[flat].mean().item(),
  }

# ── COCO mask → 14×14 patch mask ────
def pixel_mask_to_patch_mask(pixel_mask_224, threshold=0.3):
  """
  Args: pixel_mask_224: [224,224] bool array
  Returns: [14,14] BoolTensor
  """
  p = 16  # patch size
  out = torch.zeros(14, 14, dtype=torch.bool)
  for i in range(14):
      for j in range(14):
          patch = pixel_mask_224[i*p:(i+1)*p, j*p:(j+1)*p]
          if patch.mean() > threshold:
              out[i, j] = True
  return out

# ── H1 runner: process N images, return aggregate statistics ───
def run_h1(model, image_paths, masks_14x14, layer=11, device='cpu'):
  """
  On Kaggle:
      image_paths  = list of paths to ImageNet val images (JPEG)
      masks_14x14  = list of [14,14] BoolTensors from COCO annotations
  """
  results = []
  model.to(device).eval()
  for idx, (path, mask) in enumerate(zip(image_paths, masks_14x14)):
      pv = load_image(path).to(device)
      A  = extract_attention(model, pv, layer)
      H_i, Var_i, C_i = compute_routing_signals(A)
      r = measure_h1(H_i[0].cpu(), mask)
      if r is None: 
        continue
      flat = mask.flatten()
      r["cls_gap"] = (C_i[0][flat].mean() - C_i[0][~flat].mean()).item()
      results.append(r)
      if (idx+1) % 100 == 0: 
        print(f"  {idx+1}/{len(image_paths)}")
  gaps = [r["gap"] for r in results]
  return {
      "n":           len(results),
      "mean_H_gap":  np.mean(gaps),
      "std_H_gap":   np.std(gaps),
      "pct_pos":     np.mean([g > 0 for g in gaps]) * 100,
      "mean_CLS_gap":np.mean([r["cls_gap"] for r in results]),
      "per_image":   results,
  }

# ── Table 1 printer ──────
def print_h1_table(results_by_backbone):
  """Pass dict: backbone_name → run_h1() output."""
  print("\n" + "="*50)
  print(f"{'Backbone':<18} {'N':>5} {'Mean H gap':>12} {'±':>8} {'%pos':>7} {'CLS gap':>10}")
  print("-"*52)
  for name, s in results_by_backbone.items():
      print(f"{name:<18} {s['n']:>5} {s['mean_H_gap']:>12.4f} "
            f"{s['std_H_gap']:>8.4f} {s['pct_pos']:>6.1f}% {s['mean_CLS_gap']:>10.4f}")
  print("="*50)
  print("H gap > 0: background > foreground entropy ✓")
  print("CLS gap > 0: foreground more similar to CLS ✓")
  print("Best backbone = highest mean H gap + highest %pos + no artifact tokens")

# ── Visualization ─────
def plot_entropy_map(img_np, entropy_196, title="", save_path=None):
  """img_np: [224,224,3], entropy_196: [196] or [14,14]."""
  grid = torch.tensor(entropy_196).reshape(14,14)
  up = F.interpolate(grid.unsqueeze(0).unsqueeze(0).float(),
                         (224,224), mode='bilinear', align_corners=False).squeeze().numpy()
  fig, ax = plt.subplots(1, 2, figsize=(8, 4))
  ax[0].imshow(img_np); ax[0].axis('off'); ax[0].set_title("Image")
  im = ax[1].imshow(up, cmap='coolwarm_r')
  ax[1].axis('off'); ax[1].set_title("Entropy (warm=low/FG, cool=high/BG)")
  plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
  if title: 
    fig.suptitle(title)
  plt.tight_layout()
  if save_path: 
    plt.savefig(save_path, dpi=150, bbox_inches='tight'); 
    plt.close()
  else: plt.show()


# ── Quick sanity test ─────
if __name__ == "__main__":
    from transformers import ViTModel, ViTConfig
    config = ViTConfig(hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
                       intermediate_size=3072, image_size=224, patch_size=16,
                       num_channels=3, attn_implementation="eager")
    model = ViTModel(config); model.eval()

    dummy = torch.randn(2, 3, 224, 224)
    A = extract_attention(model, dummy)
    H_i, Var_i, C_i = compute_routing_signals(A)
    print(f"H_i: {H_i.shape}  Var_i: {Var_i.shape}  C_i: {C_i.shape}")

    mask = torch.zeros(14,14,dtype=torch.bool); mask[3:11,3:11] = True
    r = measure_h1(H_i[0], mask)
    print(f"H1 result (random model, no signal expected): {r}")

    import numpy as np
    dummy_np = np.random.rand(224, 224, 3).astype(np.float32)
    plot_entropy_map(dummy_np, H_i[0].numpy(),
                     title="Sanity check",
                     save_path="/mnt/user-data/outputs/h1_sanity.png")
    print("✓ make_figure1_table1.py ready")
