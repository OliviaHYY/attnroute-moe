# ══════════════════════════════════════════════════════════════════════════════
# SEEDS 1 & 2  +  E17 SPECIALIZATION
#
# Part A — Multi-seed for E2 and E4
# Part B — E17 expert specialization post-hoc analysis
#
# run seeds 1 and 2 for E2 and E4
# Part A: compare all seeds (after all 4 runs above complete)
# Part B: E17 specialization (load best E4 checkpoint)
# ══════════════════════════════════════════════════════════════════════════════

import torch
import argparse, json, math, os, time, sys
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTModel, ViTConfig

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Seeding
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    print(f"  Seed = {seed}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Inline model classes (self-contained)
# ─────────────────────────────────────────────────────────────────────────────

class ExpertFFN(nn.Module):
    def __init__(self, d=768):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d,d*4), nn.GELU(), nn.Linear(d*4,d))
    def forward(self, x): return self.net(x)

class AttentionPrior(nn.Module):
    def __init__(self, E=4, eps=1e-8):
        super().__init__()
        self.E=E; self.eps=eps
        init=torch.zeros(E,3)
        for j in range(min(E,3)): init[j,j]=1.0
        if E>3: init[3]=torch.tensor([1/3,1/3,1/3])
        self.prototypes=nn.Parameter(init)
        self.register_buffer('proto_init',init.clone())
    def forward(self,A):
        Ap=A[:,:,1:,:]; eps=self.eps
        Hi=-(Ap*(Ap+eps).log()).sum(-1).mean(1)
        Vi=((Ap-Ap.mean(1,keepdim=True))**2).mean(-1).mean(1)
        Ac=F.normalize(A[:,:,0,:].mean(1),dim=-1)
        Ci=(F.normalize(Ap.mean(1),dim=-1)*Ac.unsqueeze(1)).sum(-1)
        fi=torch.stack([-Hi,Vi,Ci],dim=-1)
        mu=fi.mean(dim=[0,1],keepdim=True); std=fi.std(dim=[0,1],keepdim=True)+eps
        return (fi-mu)/std @ self.prototypes.T.to(fi.dtype)

class StandardMoEBlock(nn.Module):
    """V-MoE baseline (E2)"""
    def __init__(self, d=768, E=4, k=2, lb=0.01):
        super().__init__()
        self.E=E; self.k=k; self.lb_coeff=lb
        self.W_r=nn.Linear(d,E,bias=False)
        self.experts=nn.ModuleList([ExpertFFN(d) for _ in range(E)])
        self.norm=nn.LayerNorm(d)
    def forward(self,z,A=None,lam=None):
        zp=self.norm(z[:,1:,:]); Si=self.W_r(zp)
        tv,ti=torch.topk(Si,self.k,dim=-1); tw=torch.softmax(tv,dim=-1)
        out=torch.zeros_like(zp)
        for e in range(self.E):
            m=(ti==e); w=(tw*m.float()).sum(-1,keepdim=True)
            out+=w*self.experts[e](zp)
        ap=torch.softmax(Si,dim=-1).mean(dim=[0,1])
        lb=self.lb_coeff*self.E*(ap*ap).sum()
        o=z.clone(); o[:,1:,:]=o[:,1:,:]+out
        return o, lb
    @torch.no_grad()
    def routing_metrics(self,z,A=None,lam=None):
        Si=self.W_r(self.norm(z[:,1:,:]))
        _,ti=torch.topk(Si,self.k,dim=-1)
        c=torch.tensor([(ti==e).float().sum().item() for e in range(self.E)])
        cv=(c.std()/(c.mean()+1e-8)).item()
        p=c/c.sum(); ue=-(p*(p+1e-8).log()).sum().item()/math.log(self.E)
        return {'cv':cv,'util_entropy':ue,'expert_counts':c.int().tolist()}

class AttnRouteMoEBlock(nn.Module):
    """Version B (E4)"""
    def __init__(self, d=768, E=4, k=2, lb=0.01, gamma=0.001):
        super().__init__()
        self.E=E; self.k=k; self.lb_coeff=lb; self.gamma=gamma
        self.prior=AttentionPrior(E)
        self.experts=nn.ModuleList([ExpertFFN(d) for _ in range(E)])
        self.norm=nn.LayerNorm(d)
        self.W_r=nn.Linear(d,E,bias=False)
    def forward(self,z,A,lam=0.0):
        zp=self.norm(z[:,1:,:]); Pi=self.prior(A)
        Si=self.W_r(zp)+lam*Pi
        tv,ti=torch.topk(Si,self.k,dim=-1); tw=torch.softmax(tv,dim=-1)
        out=torch.zeros_like(zp)
        for e in range(self.E):
            m=(ti==e); w=(tw*m.float()).sum(-1,keepdim=True)
            out+=w*self.experts[e](zp)
        ap=torch.softmax(Si,dim=-1).mean(dim=[0,1])
        lb=self.lb_coeff*self.E*(ap*ap).sum()
        pr=self.gamma*(self.prior.prototypes-self.prior.proto_init).pow(2).sum()
        o=z.clone(); o[:,1:,:]=o[:,1:,:]+out
        return o, lb+pr
    @torch.no_grad()
    def routing_metrics(self,z,A,lam=0.0):
        zp=self.norm(z[:,1:,:]); Pi=self.prior(A)
        Si=self.W_r(zp)+lam*Pi
        _,ti=torch.topk(Si,self.k,dim=-1)
        c=torch.tensor([(ti==e).float().sum().item() for e in range(self.E)])
        cv=(c.std()/(c.mean()+1e-8)).item()
        p=c/c.sum(); ue=-(p*(p+1e-8).log()).sum().item()/math.log(self.E)
        return {'cv':cv,'util_entropy':ue,'expert_counts':c.int().tolist()}

class WrappedViT(nn.Module):
    def __init__(self, backbone, moe_block, num_classes=200):
        super().__init__()
        self.backbone=backbone; self.moe_block=moe_block
        self.head=nn.Linear(768,num_classes)
        for p in backbone.parameters(): p.requires_grad=False
    def forward(self,pv,lam=0.0):
        out=self.backbone(pixel_values=pv,output_attentions=True)
        z=out.last_hidden_state; A=out.attentions[-1]
        zo,lb=self.moe_block(z,A,lam=lam)
        return self.head(zo[:,0,:]), lb

class CosineAnnealSchedule:
    def __init__(self,total_steps,frac=0.5): self.T=int(total_steps*frac)
    def get_lambda(self,step):
        if step>=self.T: return 0.0
        return math.cos(math.pi*step/(2*self.T))**2


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Data and backbone
# ─────────────────────────────────────────────────────────────────────────────

def get_loaders(data_dir, batch_size=256, num_workers=8):
    mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.Resize(256), transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(), transforms.Normalize(mean, std)])
    val_tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(mean, std)])

    # Organize val directory logic stays the same...
    val_org = os.path.join(data_dir, 'val_organized')

    train_ds = datasets.ImageFolder(os.path.join(data_dir, 'train'), train_tf)
    val_ds = datasets.ImageFolder(val_org, val_tf)

    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, pin_memory=True,
                       drop_last=True, persistent_workers=True),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=True,
                       persistent_workers=True))

def load_deit(device):
    print("  Loading facebook/deit-base-patch16-224 ...")
    m=ViTModel.from_pretrained("facebook/deit-base-patch16-224",
                                output_attentions=True,
                                attn_implementation="eager")
    m.eval()
    for p in m.parameters(): p.requires_grad=False
    return m.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Training loop (minimal, reused for both E2 and E4)
# ─────────────────────────────────────────────────────────────────────────────

class Logger:
    def __init__(self,name): self.name=name; self.history=[]; self.t0=time.time()
    def log(self,epoch,tl,ta,va,cv,ue,ec,lam):
        self.history.append({'epoch':epoch,'lambda':lam,'train_loss':tl,
            'train_acc':ta,'val_acc':va,'cv':cv,'util_entropy':ue,
            'expert_counts':ec,'drift':None,
            'elapsed_min':(time.time()-self.t0)/60})
        print(f"  ep{epoch:3d} | λ={lam:.3f} | loss={tl:.4f} | "
              f"acc={va*100:.2f}% | CV={cv:.4f}")
    def save(self,path):
        with open(path,'w') as f: json.dump({'name':self.name,'history':self.history},f,indent=2)
        print(f"  Log → {path}")

def train_epoch(model, loader, opt, sched, step, device):
    model.train()
    tl = tc = tn = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        lam = sched.get_lambda(step)

        opt.zero_grad()

        # Execute forward pass with Ampere Tensor Cores using bfloat16
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits, lb = model(imgs, lam=lam)
            loss = F.cross_entropy(logits, labels) + lb

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        tl += loss.item() * imgs.size(0)
        tc += (logits.argmax(1) == labels).sum().item()
        tn += imgs.size(0)
        step += 1

    return tl / tn, tc / tn, step

@torch.no_grad()
def val_epoch(model, loader, moe_block, sched, step, device):
    model.eval()
    tc = tn = 0
    metrics = []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        lam = sched.get_lambda(step)

        # Run the heavy backbone exactly ONCE per batch
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            out = model.backbone(pixel_values=imgs, output_attentions=True)
            z = out.last_hidden_state
            A = out.attentions[-1]

            # Pass the extracted features directly to the MoE block and head
            zo, _ = model.moe_block(z, A, lam=lam)
            logits = model.head(zo[:, 0, :])

        # Calculate accuracy
        tc += (logits.argmax(1) == labels).sum().item()
        tn += imgs.size(0)

        # Compute routing metrics using the already-computed hidden states and attention
        metrics.append(moe_block.routing_metrics(z, A, lam))

    return (tc / tn,
            np.mean([m['cv'] for m in metrics]),
            np.mean([m['util_entropy'] for m in metrics]),
            np.sum([m['expert_counts'] for m in metrics], axis=0).tolist())


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Part A: seed_run
# ─────────────────────────────────────────────────────────────────────────────

def seed_run(exp_id, seed, data_dir, out_dir, epochs, device):
    """
    Run E2 or E4 with a specific seed.
    Seed 0 = your existing logs (don't re-run).
    Run seeds 1 and 2 only.
    """

    assert seed in (1, 2), "Only run seeds 1 and 2; seed 0 = existing logs"

    set_seed(seed)
    print(f"\nRunning {exp_id} seed={seed} for {epochs} epochs")

    backbone = load_deit(device)
    train_loader, val_loader = get_loaders(data_dir)
    total_steps = epochs * len(train_loader)

    if exp_id == 'E2':
        moe   = StandardMoEBlock(E=4, k=2, lb=0.01).to(device)
        sched = CosineAnnealSchedule(total_steps, frac=0.001)
        name  = f'E2_VMoE_TinyIN_E4k2_seed{seed}'
    elif exp_id == 'E4':
        moe   = AttnRouteMoEBlock(E=4, k=2, lb=0.01, gamma=0.001).to(device)
        sched = CosineAnnealSchedule(total_steps, frac=0.50)
        name  = f'E4_VersionB_TinyIN_E4k2_T50_seed{seed}'
    else:
        raise ValueError(f"Only E2 or E4; got {exp_id}")

    model = WrappedViT(backbone, moe, num_classes=200).to(device)
    opt   = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                               lr=1e-4, weight_decay=0.05)
    lr_s  = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=total_steps, eta_min=1e-6)
    log   = Logger(name)
    step  = 0; best_acc = 0.0
    ckpt  = os.path.join(out_dir, f"{name}_best.pt")

    for epoch in range(1, epochs+1):
        tl, ta, step = train_epoch(model, train_loader, opt, sched, step, device)
        lr_s.step()
        va, cv, ue, ec = val_epoch(model, val_loader, moe, sched, step, device)
        lam = sched.get_lambda(step)
        log.log(epoch, tl, ta, va, cv, ue, ec, lam)
        if va > best_acc:
            best_acc = va
            torch.save({'epoch':epoch,'val_acc':best_acc,'state':model.state_dict()}, ckpt)
        if epoch % 10 == 0:
            print(f"  [keepalive] ep{epoch}/{epochs} best={best_acc*100:.2f}%")

    log.save(os.path.join(out_dir, f"{name}_log.json"))
    log_summary(log)
    return log

def log_summary(log):
    h=log.history; f=h[-1]; ep1=h[0]
    print(f"\n{'='*50}\n  {log.name}")
    print(f"  CV@ep1={ep1['cv']:.4f}  best_acc={max(e['val_acc'] for e in h)*100:.2f}%")
    print(f"  time={f['elapsed_min']:.0f} min\n{'='*50}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Part A: seeds_compare
# ─────────────────────────────────────────────────────────────────────────────

def seeds_compare(log_dir, out_dir):
    """
    Load all three seeds for E2 and E4 and compute mean ± std.

    Seed-0 log filenames (your existing logs — may differ from the seeded format):
        E2_VMoE_TinyImageNet_E4k2_log.json
        E4_VersionB_TinyImageNet_E4k2_T50_log.json

    Seeds 1 and 2 filenames (produced by seed_run above):
        E2_VMoE_TinyIN_E4k2_seed1_log.json
        E4_VersionB_TinyIN_E4k2_T50_seed1_log.json
        ...etc.

    The function handles both naming conventions automatically.
    """
    def try_load(candidates):
        for fname in candidates:
            path = os.path.join(log_dir, fname)
            if os.path.exists(path):
                with open(path) as f: return json.load(f)['history']
        return None

    stats = {}
    for exp_id, seed0_names, seed_prefix in [
        ('E2',
         ['E2_VMoE_TinyImageNet_E4k2_log.json',
          'E2_VMoE_TinyIN_E4k2_seed0_log.json'],
         'E2_VMoE_TinyIN_E4k2'),
        ('E4',
         ['E4_VersionB_TinyImageNet_E4k2_T50_log.json',
          'E4_VersionB_TinyIN_E4k2_T50_seed0_log.json'],
         'E4_VersionB_TinyIN_E4k2_T50'),
    ]:
        cv1s=[]; cv5s=[]; accs=[]

        # seed 0 — try both naming conventions
        h0 = try_load(seed0_names)
        if h0:
            cv1s.append(h0[0]['cv'])
            cv5s.append(h0[4]['cv'] if len(h0)>4 else None)
            accs.append(max(e['val_acc'] for e in h0)*100)
        else:
            print(f"  ✗ Seed 0 log not found for {exp_id}")

        # seeds 1 and 2
        for seed in [1, 2]:
            h = try_load([f"{seed_prefix}_seed{seed}_log.json"])
            if h:
                cv1s.append(h[0]['cv'])
                cv5s.append(h[4]['cv'] if len(h)>4 else None)
                accs.append(max(e['val_acc'] for e in h)*100)
            else:
                print(f"  ✗ Seed {seed} log not found for {exp_id}")

        if cv1s:
            stats[exp_id] = {
                'cv1_mean': np.mean(cv1s),   'cv1_std': np.std(cv1s),
                'cv5_mean': np.mean([c for c in cv5s if c is not None]),
                'acc_mean': np.mean(accs),   'acc_std': np.std(accs),
                'n':        len(cv1s),
                'cv1_vals': cv1s,
            }

    # ── print results ──────────────────────────────────────────────────────────
    print(f"\n{'═'*68}")
    print("MULTI-SEED RESULTS  (for Table 2 ± std columns)")
    print(f"{'═'*68}")
    print(f"{'Method':<22} {'n':>2} {'Acc (%)':>14} {'CV@ep1':>16}")
    print(f"{'─'*68}")
    for exp_id, r in stats.items():
        name = 'V-MoE (E2)' if exp_id=='E2' else 'AttnRoute-MoE B (E4)'
        print(f"  {name:<20} {r['n']:>2}  "
              f"{r['acc_mean']:.2f} ± {r['acc_std']:.2f}   "
              f"{r['cv1_mean']:.4f} ± {r['cv1_std']:.4f}")
    print(f"{'═'*68}")

    if 'E2' in stats and 'E4' in stats:
        r2, r4 = stats['E2'], stats['E4']
        reduction = (r2['cv1_mean']-r4['cv1_mean'])/r2['cv1_mean']*100
        print(f"\n  CV@ep1 reduction (VB over VMoE): {reduction:.1f}%")

        # significance: is the difference > 2× combined std?
        combined_std = math.sqrt(r2['cv1_std']**2 + r4['cv1_std']**2)
        diff = r2['cv1_mean'] - r4['cv1_mean']
        z_score = diff / (combined_std + 1e-8)
        print(f"  Difference:    {diff:.4f}")
        print(f"  Combined std:  {combined_std:.4f}")
        print(f"  Z-score:       {z_score:.1f}  "
              f"({'significant' if z_score>2 else 'marginal'} at n={r2['n']})")

        print(f"\n  Paste into Table 2 rows:")
        print(f"  V-MoE:  {r2['acc_mean']:.2f}$\\pm${r2['acc_std']:.2f}  "
              f"CV@ep1={r2['cv1_mean']:.4f}$\\pm${r2['cv1_std']:.4f}")
        print(f"  VB:     {r4['acc_mean']:.2f}$\\pm${r4['acc_std']:.2f}  "
              f"CV@ep1={r4['cv1_mean']:.4f}$\\pm${r4['cv1_std']:.4f}")

    # save summary JSON
    summary_path = os.path.join(out_dir, 'multi_seed_summary.json')
    with open(summary_path,'w') as f: json.dump(stats, f, indent=2)
    print(f"\n  Summary → {summary_path}")
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Part B: E17 expert specialization
# ─────────────────────────────────────────────────────────────────────────────

def e17_specialization(ckpt_path, data_dir, out_dir, device, n_bins=10):
    """
    Post-hoc specialization analysis on best Version B checkpoint.

    What this produces:
        figure5_specialization.png   ← Figure 5 of the paper

    What to look for:
        Non-flat bars across entropy deciles → prior shaped expert specialization
        Flat bars → routing converged to semantically undifferentiated pattern

    The checkpoint must be the E4 best.pt file saved during training.
    It contains the full model state dict (backbone + moe_block + head).
    """
    print(f"\nRunning E17 specialization analysis ...")
    print(f"  Checkpoint: {ckpt_path}")

    # ── load model ─────────────────────────────────────────────────────────────
    backbone  = load_deit(device)
    moe_block = AttnRouteMoEBlock(E=4, k=2, lb=0.01, gamma=0.001).to(device)
    model     = WrappedViT(backbone, moe_block, num_classes=200).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['state'])
    model.eval()
    print(f"  Loaded checkpoint: epoch={ckpt['epoch']}  val_acc={ckpt['val_acc']*100:.2f}%")

    # ── validation loader (no augmentation) ────────────────────────────────────
    mean=[0.485,0.456,0.406]; std=[0.229,0.224,0.225]
    val_tf=transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(mean,std)])
    val_org=os.path.join(data_dir,'val_organized')
    val_ds =datasets.ImageFolder(val_org,val_tf)
    val_loader=DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=4)
    print(f"  Val dataset: {len(val_ds):,} images")

    # ── collect routing assignments and entropy values ──────────────────────────
    all_H    = []   # per-token entropy (from attention maps)
    all_asgn = []   # top-1 expert assignment per token

    with torch.no_grad():
        for batch_idx, (imgs, _) in enumerate(val_loader):
            imgs = imgs.to(device)
            out  = backbone(pixel_values=imgs, output_attentions=True)
            z    = out.last_hidden_state
            A    = out.attentions[-1]

            # compute entropy
            Ap  = A[:,:,1:,:]
            Hi  = -(Ap*(Ap+1e-8).log()).sum(-1).mean(1)   # [B, 196]

            # routing at lambda=0 (inference mode)
            Pi  = moe_block.prior(A)
            zp  = moe_block.norm(z[:,1:,:])
            Si  = moe_block.W_r(zp)   # λ=0 at inference
            _, ti = torch.topk(Si, 1, dim=-1)  # top-1 only [B, 196, 1]

            all_H.extend(Hi.flatten().cpu().tolist())
            all_asgn.extend(ti.squeeze(-1).flatten().cpu().tolist())

            if (batch_idx+1) % 10 == 0:
                print(f"  Processed {(batch_idx+1)*256:,} / {len(val_ds):,} tokens ...")

    all_H    = np.array(all_H)
    all_asgn = np.array(all_asgn)
    E        = moe_block.E

    print(f"  Total tokens: {len(all_H):,}")
    print(f"  Entropy range: [{all_H.min():.3f}, {all_H.max():.3f}]")
    for e in range(E):
        pct = (all_asgn==e).mean()*100
        print(f"  Expert {e}: {pct:.1f}% of tokens (global)")

    # ── bin by entropy decile and compute routing fractions ────────────────────
    bins  = np.percentile(all_H, np.linspace(0, 100, n_bins+1))
    fracs = np.zeros((n_bins, E))
    counts_per_bin = []

    for b in range(n_bins):
        lo = bins[b]; hi = bins[b+1]
        mask = (all_H >= lo) & (all_H <= hi)
        counts_per_bin.append(mask.sum())
        if mask.sum() == 0: continue
        for e in range(E):
            fracs[b, e] = (all_asgn[mask] == e).mean()

    # ── figure ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(n_bins)
    w = 0.8 / E
    colors = plt.cm.Set1(np.linspace(0, 0.85, E))

    for e in range(E):
        ax.bar(x + e*w, fracs[:, e], w,
               label=f'Expert {e}', color=colors[e], alpha=0.85, zorder=3)

    ax.set_xlabel(
        'Token entropy decile\n'
        'D1 = lowest entropy (focused/foreground)  →  '
        'D10 = highest entropy (diffuse/background)',
        fontsize=10
    )
    ax.set_ylabel('Fraction of tokens routed here (top-1)', fontsize=10)
    ax.set_title(
        'Figure 5 — Expert routing fraction vs. token entropy decile\n'
        'Non-flat profiles indicate the prior shaped expert specialization '
        'along the semantic axis it was designed to exploit.',
        fontsize=10
    )
    ax.set_xticks(x + 0.8/2)
    ax.set_xticklabels([f'D{i+1}' for i in range(n_bins)], fontsize=9)
    ax.axhline(1/E, color='gray', ls='--', lw=1, alpha=0.6,
               label=f'Uniform (1/{E})')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(0, 0.5); ax.grid(axis='y', alpha=0.3, zorder=0)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, 'figure5_specialization.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Figure 5 → {fig_path}")

    # ── monotonicity test: is at least one expert monotone with entropy? ─────────
    mono_experts = []
    for e in range(E):
        diffs = np.diff(fracs[:, e])
        # monotone increasing or decreasing across ≥7 of 9 consecutive pairs
        if (diffs > 0).sum() >= 7 or (diffs < 0).sum() >= 7:
            mono_experts.append(e)

    print(f"\n  Monotone experts (routing fraction trends with entropy): {mono_experts}")
    if mono_experts:
        print(f"  ✓ Prior shaped specialization — non-flat routing along entropy axis")
        print(f"    Expert(s) {mono_experts} show monotone entropy response")
    else:
        print(f"  Routing fractions are roughly flat across entropy deciles")
        print(f"  → Report honestly: 'training converged to semantically")
        print(f"    undifferentiated routing despite the prior initialization.'")
        print(f"  → This is not a failure; mention in §6 as a limitation")

    # save raw data
    data_path = os.path.join(out_dir, 'e17_specialization_data.npz')
    np.savez(data_path, fracs=fracs, bins=bins, counts=np.array(counts_per_bin))
    print(f"  Raw data → {data_path}")

    return fracs, bins


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['seed_run','seeds_compare','e17'],
                        default='e17')
    # seed_run args
    parser.add_argument('--exp',      default='E4', help='E2 or E4')
    parser.add_argument('--seed',     type=int, default=2, help='1 or 2 only')
    parser.add_argument('--epochs',   type=int, default=50)
    parser.add_argument('--data_dir', default='/content/tiny-imagenet-200', help='Local fast disk storage')
    parser.add_argument('--out_dir',  default='/content/drive/MyDrive/tinyimagenet_experiments', help='Google Drive persistent storage')
    # e17 args
    parser.add_argument('--ckpt',     default='/content/drive/MyDrive/tinyimagenet_experiments/E4_VersionB_TinyIN_T50_seed2_best.pt',
                        help='path to best E4 checkpoint .pt file')
    parser.add_argument('--n_bins',   type=int, default=10)
    args, _ = parser.parse_known_args()

    # auto_setup_environment(args.data_dir, args.out_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda': print(f"GPU: {torch.cuda.get_device_name(0)}")

    if args.mode == 'seed_run':
        # Epoch count: match original runs
        # E2 (V-MoE) ran 50 epochs; E4 ran 45 epochs
        epochs = 50 if args.exp == 'E2' else 45
        seed_run(args.exp, args.seed, args.data_dir, args.out_dir,
                 epochs=epochs, device=device)

    elif args.mode == 'seeds_compare':
        seeds_compare(log_dir=args.out_dir, out_dir=args.out_dir)

    elif args.mode == 'e17':
        target_ckpt = args.ckpt if args.ckpt else os.path.join(args.out_dir, "E4_VersionB_TinyIN_T50_seed2_best.pt")
        assert os.path.exists(target_ckpt), f"❌ Specified checkpoint missing: {target_ckpt}"
        e17_specialization(target_ckpt, args.data_dir, args.out_dir, device, n_bins=args.n_bins)
