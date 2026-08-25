"""
make_figure5b.py — Generate Figure 5b: Random Prior specialization comparison
Run with the Random Prior (E10b) best checkpoint:

Also generates a side-by-side comparison of Figure 5 (Version B) and Figure 5b
(Random Prior) as figure5_comparison.png for a single combined figure.

The Random Prior model uses version='b' in AttnRouteMoEBlock (fixed random prior,
no AttentionPrior learned module). This script handles both checkpoint types.
"""

import argparse, json, os, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTModel


class ExpertFFN(nn.Module):
    def __init__(self, d=768):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d,d*4), nn.GELU(), nn.Linear(d*4,d))
    def forward(self, x): return self.net(x)

class AttentionPrior(nn.Module):
    def __init__(self, E=4, eps=1e-8):
        super().__init__()
        self.E = E; self.eps = eps
        init = torch.zeros(E, 3)
        for j in range(min(E, 3)): init[j, j] = 1.0
        if E > 3: init[3] = torch.tensor([1/3, 1/3, 1/3])
        self.prototypes = nn.Parameter(init)
        self.register_buffer('proto_init', init.clone())
    def forward(self, A):
        Ap = A[:,:,1:,:]
        Hi = -(Ap*(Ap+self.eps).log()).sum(-1).mean(1)
        Vi = ((Ap-Ap.mean(1,keepdim=True))**2).mean(-1).mean(1)
        Ac = F.normalize(A[:,:,0,:].mean(1), dim=-1)
        Ci = (F.normalize(Ap.mean(1), dim=-1)*Ac.unsqueeze(1)).sum(-1)
        fi = torch.stack([-Hi, Vi, Ci], dim=-1)
        mu = fi.mean(dim=[0,1], keepdim=True)
        std = fi.std(dim=[0,1], keepdim=True) + 1e-6
        return (fi-mu)/std @ self.prototypes.T

class AttnRouteMoEBlock(nn.Module):
    def __init__(self, d=768, E=4, k=2, version='B', lb=0.01, gamma=0.001, seed=42):
        super().__init__()
        self.E = E; self.k = k; self.version = version
        self.experts = nn.ModuleList([ExpertFFN(d) for _ in range(E)])
        self.norm = nn.LayerNorm(d)
        if version in ('B', 'b'):
            self.W_r = nn.Linear(d, E, bias=False)
        if version == 'b':
            g = torch.Generator(); g.manual_seed(seed)
            rand_p = torch.randn(196, E, generator=g)
            rand_p = rand_p * (1/math.sqrt(E)) / rand_p.std()
            self.register_buffer('rand_prior', rand_p)
        else:
            self.prior = AttentionPrior(E)
    def forward(self, z, A, lam=0.0):
        zp = self.norm(z[:,1:,:])
        Pi = self.rand_prior.unsqueeze(0) if self.version=='b' else self.prior(A)
        Si = self.W_r(zp) + lam*Pi if self.version in ('B','b') else Pi
        tv,ti = torch.topk(Si, self.k, dim=-1); tw = torch.softmax(tv,dim=-1)
        out = torch.zeros_like(zp)
        for e in range(self.E):
            m=(ti==e); w=(tw*m.float()).sum(-1,keepdim=True)
            out += w*self.experts[e](zp)
        ap = torch.softmax(Si,dim=-1).mean(dim=[0,1])
        lb_loss = 0.01*self.E*(ap*ap).sum()
        o = z.clone(); o[:,1:,:] = o[:,1:,:]+out
        return o, lb_loss

class WrappedViT(nn.Module):
    def __init__(self, backbone, moe_block, num_classes=200):
        super().__init__()
        self.backbone = backbone; self.moe_block = moe_block
        self.head = nn.Linear(768, num_classes)
        for p in backbone.parameters(): p.requires_grad = False
    def forward(self, pv, lam=0.0):
        out = self.backbone(pixel_values=pv, output_attentions=True)
        z = out.last_hidden_state; A = out.attentions[-1]
        zo, lb = self.moe_block(z, A, lam=lam)
        return self.head(zo[:,0,:]), lb


# ── Analysis function ───────

def compute_entropy_fracs(backbone, moe_block, val_loader, device, n_bins=10,
                          use_attention_prior=True):
    """
    Compute routing fraction per expert per entropy decile.
    use_attention_prior=True  → read from moe_block.prior (Version B)
    use_attention_prior=False → use moe_block.rand_prior (Random Prior)
    Returns: fracs [n_bins, E], bins [n_bins+1]
    """
    all_H = []; all_asgn = []
    moe_block.eval()
    with torch.no_grad():
        for imgs, _ in val_loader:
            imgs = imgs.to(device)
            out = backbone(pixel_values=imgs, output_attentions=True)
            z = out.last_hidden_state; A = out.attentions[-1]
            Ap = A[:,:,1:,:]
            Hi = -(Ap*(Ap+1e-8).log()).sum(-1).mean(1)   # [B, 196]
            zp = moe_block.norm(z[:,1:,:])
            if use_attention_prior:
                Pi = moe_block.prior(A)
            else:
                Pi = moe_block.rand_prior.unsqueeze(0)
            Si = moe_block.W_r(zp)   # λ=0 at inference
            _, ti = torch.topk(Si, 1, dim=-1)
            all_H.extend(Hi.flatten().cpu().tolist())
            all_asgn.extend(ti.squeeze(-1).flatten().cpu().tolist())
    all_H = np.array(all_H); all_asgn = np.array(all_asgn)
    E = moe_block.E
    bins = np.percentile(all_H, np.linspace(0, 100, n_bins+1))
    fracs = np.zeros((n_bins, E))
    for b in range(n_bins):
        mask = (all_H >= bins[b]) & (all_H <= bins[b+1])
        if mask.sum() == 0: continue
        for e in range(E):
            fracs[b, e] = (all_asgn[mask] == e).mean()
    return fracs, bins


def plot_fracs(fracs, title, ax, E=4, n_bins=10):
    x = np.arange(n_bins); w = 0.8/E
    colors = plt.cm.Set1(np.linspace(0, 0.85, E))
    for e in range(E):
        ax.bar(x+e*w, fracs[:,e], w, label=f'Expert {e}',
               color=colors[e], alpha=0.85, zorder=3)
    ax.axhline(1/E, color='gray', ls='--', lw=1.2, alpha=0.7, label=f'Uniform (1/{E})')
    ax.set_xticks(x + 0.4)
    ax.set_xticklabels([f'D{i+1}' for i in range(n_bins)], fontsize=8)
    ax.set_ylim(0, 0.55); ax.set_ylabel('Fraction of tokens (top-1)', fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, zorder=0)
    ax.legend(loc='upper right', fontsize=8)


def run(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load backbone
    print("Loading DeiT backbone...")
    backbone = ViTModel.from_pretrained("facebook/deit-base-patch16-224",
                                        output_attentions=True,
                                        attn_implementation="eager")
    backbone.eval().to(device)
    for p in backbone.parameters(): p.requires_grad = False

    # Val loader
    mean=[0.485,0.456,0.406]; std=[0.229,0.224,0.225]
    val_tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                  transforms.ToTensor(), transforms.Normalize(mean,std)])
    import shutil
    val_org = os.path.join(args.data_dir, 'val_organized')
    if not os.path.exists(val_org):
        ann = os.path.join(args.data_dir, 'val', 'val_annotations.txt')
        with open(ann) as f:
            for line in f:
                p = line.strip().split('\t')
                src = os.path.join(args.data_dir,'val','images',p[0])
                dst = os.path.join(val_org, p[1])
                os.makedirs(dst, exist_ok=True)
                shutil.copy(src, os.path.join(dst, p[0]))
    val_loader = DataLoader(datasets.ImageFolder(val_org, val_tf),
                            batch_size=256, shuffle=False, num_workers=4)

    # ── Load Version B checkpoint ────────────────────────────────────────────
    print(f"\nLoading Version B checkpoint: {args.ckpt_vb}")
    moe_vb = AttnRouteMoEBlock(E=4, k=2, version='B').to(device)
    model_vb = WrappedViT(backbone, moe_vb).to(device)
    ckpt_vb = torch.load(args.ckpt_vb, map_location=device)
    model_vb.load_state_dict(ckpt_vb['state'])
    print(f" Version B: epoch={ckpt_vb['epoch']}  val_acc={ckpt_vb['val_acc']*100:.2f}%")

    print("\nComputing Version B entropy fracs...")
    fracs_vb, bins = compute_entropy_fracs(backbone, moe_vb, val_loader, device,
                                            use_attention_prior=True)

    # ── Load Random Prior checkpoint ───────
    print(f"\nLoading Random Prior checkpoint: {args.ckpt_rp}")
    moe_rp = AttnRouteMoEBlock(E=4, k=2, version='b', seed=42).to(device)
    model_rp = WrappedViT(backbone, moe_rp).to(device)
    ckpt_rp = torch.load(args.ckpt_rp, map_location=device)

    # Name mismatch of head. and classifier.
    if isinstance(ckpt_rp, dict) and 'state' in ckpt_rp:
      raw_state_dict = ckpt_rp['state']
    elif isinstance(ckpt_rp, dict) and 'state_dict' in ckpt_rp:
      raw_state_dict = ckpt_rp['state_dict']
    elif isinstance(ckpt_rp, dict) and 'model' in ckpt_rp:
      raw_state_dict = ckpt_rp['model']
    else:
      raw_state_dict = ckpt_rp

    new_state_dict = {}
    for k, v in raw_state_dict.items():
      new_key = k.replace('classifier.', 'head.')
      new_state_dict[new_key] = v

    model_rp.load_state_dict(new_state_dict)
    print(f" Random Prior: epoch={ckpt_rp['epoch']}  val_acc={ckpt_rp['val_acc']*100:.2f}%")

    print("\nComputing Random Prior entropy fracs...")
    fracs_rp, _ = compute_entropy_fracs(backbone, moe_rp, val_loader, device,
                                         use_attention_prior=False)

    # ── Plot side-by-side comparison ─────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        'Expert routing fraction vs. token entropy decile\n'
        'D1 = lowest entropy (foreground)  →  D10 = highest entropy (background)',
        fontsize=10
    )
    plot_fracs(fracs_vb, '(a) Version B (semantic prior)', axes[0])
    plot_fracs(fracs_rp, '(b) Random Prior (no semantic content)', axes[1])
    axes[0].set_xlabel('Token entropy decile', fontsize=9)
    axes[1].set_xlabel('Token entropy decile', fontsize=9)
    plt.tight_layout()

    path_side = os.path.join(args.out_dir, 'figure5_comparison.png')
    plt.savefig(path_side, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n→ Side-by-side figure: {path_side}")

    # ── Also save Figure 5b alone ───────
    fig2, ax2 = plt.subplots(figsize=(11, 4.5))
    plot_fracs(fracs_rp, 'Random Prior — expert routing fraction vs. entropy decile', ax2)
    ax2.set_xlabel(
        'Token entropy decile\n'
        'D1 = lowest (foreground)  →  D10 = highest (background)', fontsize=9)
    plt.tight_layout()
    path_5b = os.path.join(args.out_dir, 'figure5b_specialization_compare.png')
    plt.savefig(path_5b, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"→ Figure 5b alone: {path_5b}")

    # ── Print key stats ────────
    print("\nVersion B max fraction in D10:", fracs_vb[9].max())
    print("Random Prior max fraction in D10:", fracs_rp[9].max())
    print("VB expert 3 D10:", fracs_vb[9, 3])
    print("RP expert 3 D10:", fracs_rp[9, 3])
    diff = fracs_vb[9, 3] - fracs_rp[9, 3]
    print(f"Difference in D10 Expert 3: VB-RP = {diff:+.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_vb', required=True,
            default='/content/drive/MyDrive/tinyimagenet_experiments/E4_VersionB_TinyIN_T50_seed1_best.pt',
            help='Version B best checkpoint (E4_VersionB...seed0_best.pt)')
    parser.add_argument('--ckpt_rp', required=True,
            default='/content/drive/MyDrive/tinyimagenet_experiments/E10b_RandomPrior_TinyIN_T50_best.pt',
            help='Random Prior best checkpoint (E10b_RandomPrior...best.pt)')
    parser.add_argument('--data_dir',default='/content/tiny-imagenet-200')
    parser.add_argument('--out_dir', default='/content/outputs')
    args, _ = parser.parse_known_args(args=[
    '--ckpt_vb', '/content/drive/MyDrive/tinyimagenet_experiments/E4_VersionB_TinyIN_T50_seed1_best.pt',
    '--ckpt_rp', '/content/drive/MyDrive/tinyimagenet_experiments/E10b_RandomPrior_TinyIN_T50_best.pt'
    ])
    os.makedirs(args.out_dir, exist_ok=True)
    run(args)
